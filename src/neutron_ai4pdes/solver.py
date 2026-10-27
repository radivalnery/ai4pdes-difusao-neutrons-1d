"""Iteracao de potencia e diagnosticos; base v4.1 preservada."""
import platform
import time
import numpy as np
import torch
from .models import OperadorDifusao1D, SolverFonteFixaUNet1D, resolver_tridiagonal_thomas
from .fem import OperadorMEF1D
from .references import calcular_k_eff_analitico, fluxo_analitico_homogeneo
from .metrics import erro_forma_fluxo

class SolverDifusaoAI4PDEs:
    REFERENCIAS = {
        'homogeneo': {'k_eff': None, 'fonte': 'Analítica'},
        'heterogeneo': {'k_eff': 1.09506, 'fonte': 'Couto (2003)'}
    }
    
    def __init__(self, L, N, materiais, cond_esquerda='reflexiva', cond_direita='vácuo',
                 tol_k=1e-6, tol_phi=1e-5, max_iter=1000, potencia_nominal=100.0,
                 pontos_interesse=None, progress_callback=None, dispositivo_preferido='auto',
                 omega_fonte=0.75, amortecimento_unet=0.20,
                 tol_fonte=None, max_iter_fonte=5000,
                 metodo_fonte="unet_multigrid", guardar_historicos_fonte=False,
                 permitir_thomas_comparacao=False, discretizacao='mdf', precisao='float64'):
        
        self.discretizacao = str(discretizacao).lower()
        if self.discretizacao not in ('mdf', 'mef'):
            raise ValueError('Discretizacao deve ser mdf ou mef.')
        if L <= 0 or N < 1 or int(N) != N or max_iter < 1 or max_iter_fonte < 1:
            raise ValueError('Comprimento, malha e limites de iteracoes devem ser positivos.')
        if min(tol_k, tol_phi, tol_phi if tol_fonte is None else tol_fonte) <= 0:
            raise ValueError('As tolerancias devem ser positivas.')
        if N == 1 and cond_esquerda == cond_direita == 'vácuo':
            raise ValueError('Dois contornos de fluxo nulo requerem N >= 2.')
        if precisao not in ('float32', 'float64'):
            raise ValueError('Precisao deve ser float32 ou float64.')
        self.precisao = precisao
        self.dtype = getattr(torch, precisao)
        self.L = L
        self.N = N
        self.h = L / N
        self.x = np.linspace(0, L, N+1)
        self.cond_esquerda = cond_esquerda
        self.cond_direita = cond_direita
        self.tol_k = tol_k
        self.tol_phi = tol_phi
        self.max_iter = max_iter
        self.potencia_nominal = potencia_nominal
        self.pontos_interesse = pontos_interesse or [0.0, 50.0, 100.0, 150.0]
        self.progress_callback = progress_callback
        self.dispositivo_preferido = dispositivo_preferido
        self.omega_fonte = float(omega_fonte)
        self.amortecimento_unet = float(amortecimento_unet)
        self.tol_fonte = float(tol_phi if tol_fonte is None else tol_fonte)
        self.max_iter_fonte = int(max_iter_fonte)
        self.metodo_fonte = metodo_fonte
        self.guardar_historicos_fonte = guardar_historicos_fonte
        self.permitir_thomas_comparacao = bool(permitir_thomas_comparacao)
        
        self.materiais = materiais
        self.D_arr = np.zeros(N+1)
        self.Sigma_a_arr = np.zeros(N+1)
        self.nuSigma_f_arr = np.zeros(N+1)
        
        for i, x in enumerate(self.x):
            for mat in materiais:
                if mat['inicio'] <= x <= mat['fim']:
                    self.D_arr[i] = mat['D']
                    self.Sigma_a_arr[i] = mat['Sigma_a']
                    self.nuSigma_f_arr[i] = mat['nuSigma_f']
                    break
        
        self.cuda_disponivel = bool(torch.cuda.is_available())
        if dispositivo_preferido == 'cpu':
            self.device = torch.device('cpu')
        elif dispositivo_preferido == 'gpu' and self.cuda_disponivel:
            self.device = torch.device('cuda')
        elif dispositivo_preferido == 'gpu' and not self.cuda_disponivel:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if self.cuda_disponivel else 'cpu')

        self.nome_dispositivo = self.obter_nome_dispositivo()
        self.metodo_executado = (
            "Neural Physics 1D adaptado com PyTorch CUDA"
            if self.device.type == 'cuda'
            else "Neural Physics 1D adaptado com PyTorch CPU"
        )
        self.modelo_A = None
        self.solver_fonte_fixa = None
        self.iteracoes_fonte_fixa = []
        self.residuos_fonte_fixa = []
        self.convergiu_fonte_fixa = []
        self.historico_residuo_fonte_ultima_chamada = []
        self.historicos_residuo_fonte = []
        
        self.k_eff = None
        self.phi = None
        self.phi_normalizado = None
        self.historico_k = []
        self.historico_phi = []
        self.tempos_iteracao = []
        self.tempo_total = 0.0
        self.tempo_medio_iteracao = 0.0
        self.tempos_detalhados = {
            "tempo_montagem_operador_s": 0.0,
            "tempo_montagem_hierarquia_s": 0.0,
            "tempo_montagem_total_s": 0.0,
            "tempo_transferencia_numpy_torch_s": 0.0,
            "tempo_transferencia_torch_numpy_s": 0.0,
            "tempo_transferencia_total_s": 0.0,
            "tempo_loop_iterativo_s": 0.0,
            "tempo_iteracao_numerica_s": 0.0,
            "tempo_fonte_fixa_s": 0.0,
            "tempo_atualizacao_potencia_s": 0.0,
            "tempo_total_s": 0.0,
            "tempo_medio_iteracao_s": 0.0,
        }
        
        self.iteracoes_totais = 0
        self.convergiu = False
        self.erro_k = 0.0
        self.erro_phi_iterativo = 0.0
        self.erro_phi_max = 0.0
        self.erro_phi_pontos = {}
        self.k_ref = None
        self.fonte_ref = None
        self.phi_ref = None
        self.is_homogeneo = len(materiais) == 1

    def _sincronizar(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _tensor(self, value):
        """Mede conversoes e copias separadas do custo numerico."""
        self._sincronizar()
        start = time.perf_counter()
        result = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        self._sincronizar()
        self.tempos_detalhados['tempo_transferencia_numpy_torch_s'] += time.perf_counter()-start
        return result

    def _numpy(self, value):
        self._sincronizar()
        start = time.perf_counter()
        result = value.detach().cpu().numpy()
        self._sincronizar()
        self.tempos_detalhados['tempo_transferencia_torch_numpy_s'] += time.perf_counter()-start
        return result

    def fonte_fissao(self, phi):
        """Mesmo caminho Conv1d fixo para F_MDF diagonal e F_MEF consistente."""
        if torch.is_tensor(phi):
            return self.modelo_A.aplicar_fissao(phi)
        return self._numpy(self.modelo_A.aplicar_fissao(self._tensor(phi)))

    def integral_fissao(self, phi):
        return float(np.dot(self.modelo_A.pesos_producao_numpy, phi))

    def descricao_discretizacao(self):
        if self.discretizacao == 'mef':
            formulation = 'MEF P1: A=H^-1(K+M_a), F=H^-1 M_f, massa consistente (sem lumping). '
        else:
            formulation = 'MDF conservativo: difusao com media harmonica e fissao diagonal. '
        return (formulation + 'Banco Conv1d fixo comum para A/F; mesmo ciclo V, '
                'Jacobi, P linear e R=Hc^-1 P^T Hf. Precisao: ' + self.precisao + '.')

    @staticmethod
    def nome_metodo(metodo):
        nomes = {
            "unet_multigrid": "U-Net/multigrid sem treinamento",
            "thomas": "Thomas clássico",
        }
        return nomes.get(metodo, metodo)

    def obter_nome_dispositivo(self):
        if self.device.type == 'cuda':
            try:
                return torch.cuda.get_device_name(0)
            except Exception:
                return "GPU CUDA"
        return platform.processor() or platform.machine() or "CPU"
    
    def criar_operador_A(self):
        inicio_operador = time.perf_counter()
        if self.discretizacao == 'mef':
            self.modelo_A = OperadorMEF1D(self.x, self.materiais,
                                          self.cond_esquerda, self.cond_direita, dtype=self.dtype)
        else:
            self.modelo_A = OperadorDifusao1D(self.D_arr, self.Sigma_a_arr, self.h,
                                              self.cond_esquerda, self.cond_direita,
                                              x=self.x, materiais=self.materiais,
                                              nuSigma_f_arr=self.nuSigma_f_arr, dtype=self.dtype)
        self.tempos_detalhados["tempo_montagem_operador_s"] += time.perf_counter() - inicio_operador

        inicio_hierarquia = time.perf_counter()
        self.solver_fonte_fixa = None
        if self.metodo_fonte == 'unet_multigrid':
            self.solver_fonte_fixa = SolverFonteFixaUNet1D(self.modelo_A, omega=self.omega_fonte,
                                            amortecimento_unet=self.amortecimento_unet)
        self.tempos_detalhados["tempo_montagem_hierarquia_s"] += time.perf_counter() - inicio_hierarquia
        self._sincronizar()
        inicio_transferencia = time.perf_counter()
        if self.solver_fonte_fixa is not None:
            self.solver_fonte_fixa.to(self.device)
        else:
            self.modelo_A.to(self.device)
        self._sincronizar()
        self.tempos_detalhados['tempo_transferencia_numpy_torch_s'] += time.perf_counter()-inicio_transferencia

        self.metodo_executado = (
            "Adaptação 1D Neural Physics/AI4PDEs: operadores convolucionais fixos "
            "e ciclo U-Net/multigrid geométrico sem treinamento"
        )
    
    def resolver_fonte_fixa_unet(self, S, chute=None):
        if self.solver_fonte_fixa is None:
            raise RuntimeError("O resolvedor de fonte fixa ainda não foi criado.")
        S_tensor = S if torch.is_tensor(S) else self._tensor(S)
        chute_tensor = None
        if chute is not None:
            chute_tensor = chute if torch.is_tensor(chute) else self._tensor(chute)
        resultado = self.solver_fonte_fixa.resolver(
            S_tensor,
            chute=chute_tensor,
            tol=self.tol_fonte,
            max_iter=self.max_iter_fonte,
        )
        phi = resultado.phi
        n_iter = resultado.iteracoes
        residuo = resultado.residuo
        historico = resultado.historico_residuo
        self.iteracoes_fonte_fixa.append(int(n_iter))
        self.residuos_fonte_fixa.append(float(residuo))
        self.convergiu_fonte_fixa.append(bool(resultado.convergiu))
        self.historico_residuo_fonte_ultima_chamada = list(historico)
        if self.guardar_historicos_fonte:
            self.historicos_residuo_fonte.append(list(historico))
        return phi

    def resolver_fonte_fixa_thomas(self, S):
        if not self.permitir_thomas_comparacao:
            raise RuntimeError(
                "Thomas está disponível apenas como referência numérica clássica. "
                "O método principal deve ser unet_multigrid."
            )
        if self.modelo_A is None:
            raise RuntimeError("O operador A ainda não foi criado.")
        S_np = self._numpy(S) if torch.is_tensor(S) else np.asarray(S, dtype=float)
        lower, diag, upper, rhs = self.modelo_A.matriz_tridiagonal_numpy(S_np)
        phi_np = resolver_tridiagonal_thomas(lower, diag, upper, rhs)
        phi_tensor = self._tensor(phi_np)
        residuo_tensor = self._tensor(rhs) - self.modelo_A(phi_tensor)
        residuo = float(torch.linalg.vector_norm(residuo_tensor, ord=float('inf')).item())
        norma_rhs = max(float(np.linalg.norm(rhs, ord=np.inf)), torch.finfo(self.dtype).tiny)
        self.iteracoes_fonte_fixa.append(1)
        self.residuos_fonte_fixa.append(residuo)
        self.convergiu_fonte_fixa.append(True)
        self.historico_residuo_fonte_ultima_chamada = [residuo / norma_rhs]
        if self.guardar_historicos_fonte:
            self.historicos_residuo_fonte.append([residuo / norma_rhs])
        return phi_tensor

    def resolver_fonte_fixa(self, S, chute=None, metodo=None):
        metodo = metodo or self.metodo_fonte
        if metodo == "unet_multigrid":
            return self.resolver_fonte_fixa_unet(S, chute=chute)
        if metodo == "thomas":
            return self.resolver_fonte_fixa_thomas(S)
        raise ValueError(f"Método de fonte fixa desconhecido: {metodo}")
    
    def calcular_integral(self, f):
        integral = 0.5 * (f[0] + f[-1])
        integral += np.sum(f[1:-1])
        integral *= self.h
        return integral
    
    def calcular_referencias(self):
        if self.is_homogeneo:
            D = self.D_arr[0]
            Sigma_a = self.Sigma_a_arr[0]
            nuSigma_f = self.nuSigma_f_arr[0]
            self.k_ref = calcular_k_eff_analitico(D, Sigma_a, nuSigma_f, self.L,
                                                  self.cond_esquerda, self.cond_direita)
            self.fonte_ref = 'Analítica (fórmula)'
            self.phi_ref = fluxo_analitico_homogeneo(self.x, self.L, 
                                                     self.cond_esquerda, self.cond_direita)
            if np.max(self.phi_ref) > 0:
                self.phi_ref = self.phi_ref / np.max(self.phi_ref)
        else:
            self.k_ref = self.REFERENCIAS['heterogeneo']['k_eff']
            self.fonte_ref = self.REFERENCIAS['heterogeneo']['fonte']
            self.phi_ref = None
    
    def normalizar_potencia(self, phi):
        P_atual = 3.2e-11 * self.integral_fissao(phi)
        if P_atual > 0:
            return phi * (self.potencia_nominal / P_atual)
        return phi

    def fluxo_analitico_pontos(self, pontos, potencia=False):
        """Avalia a formula no x solicitado, mesmo quando x nao e um no."""
        grid = fluxo_analitico_homogeneo(self.x, self.L, self.cond_esquerda, self.cond_direita)
        scale = np.max(grid)
        values = fluxo_analitico_homogeneo(np.asarray(pontos), self.L,
                                          self.cond_esquerda, self.cond_direita) / scale
        if potencia:
            normalized = self.normalizar_potencia(self.phi_ref)
            index = np.argmax(self.phi_ref)
            values = values * normalized[index] / self.phi_ref[index]
        values[np.abs(values) <= 1e-12 * max(np.max(np.abs(values)), 1.)] = 0.
        return values
    
    def resolver(self, metodo_fonte=None):
        if metodo_fonte is not None:
            self.metodo_fonte = metodo_fonte
        if self.metodo_fonte == 'thomas' and not self.permitir_thomas_comparacao:
            raise RuntimeError('Thomas disponivel apenas como referencia numerica classica.')
        if self.metodo_fonte not in ('unet_multigrid', 'thomas'):
            raise ValueError('Metodo de fonte desconhecido.')
        self.tempos_detalhados = {
            "tempo_montagem_operador_s": 0.0,
            "tempo_montagem_hierarquia_s": 0.0,
            "tempo_montagem_total_s": 0.0,
            "tempo_transferencia_numpy_torch_s": 0.0,
            "tempo_transferencia_torch_numpy_s": 0.0,
            "tempo_transferencia_total_s": 0.0,
            "tempo_loop_iterativo_s": 0.0,
            "tempo_iteracao_numerica_s": 0.0,
            "tempo_fonte_fixa_s": 0.0,
            "tempo_atualizacao_potencia_s": 0.0,
            "tempo_preparacao_fonte_s": 0.0,
            "tempo_total_s": 0.0,
            "tempo_medio_iteracao_s": 0.0,
        }
        tempo_total_inicio_global = time.perf_counter()
        self.criar_operador_A()
        self.calcular_referencias()
        n = self.N + 1
        phi = np.ones(n)
        if self.cond_esquerda == 'vácuo':
            phi[0] = 0.
        if self.cond_direita == 'vácuo':
            phi[-1] = 0.
        k_eff = 1.0
        self.historico_k = []
        self.historico_phi = []
        self.tempos_iteracao = []
        self.iteracoes_fonte_fixa = []
        self.residuos_fonte_fixa = []
        self.convergiu_fonte_fixa = []
        self.historico_residuo_fonte_ultima_chamada = []
        self.historicos_residuo_fonte = []
        self.convergiu = False
        self.iteracoes_totais = 0
        self.erro_phi_pontos = {}
        self.metodo_executado = (
            f"{self.nome_metodo(self.metodo_fonte)} / {self.discretizacao.upper()} aplicado ao problema de fonte fixa"
        )
        
        print("\n" + "="*60)
        print("SOLVER DE DIFUSÃO DE NÊUTRONS COM AI4PDEs 1D ADAPTADO")
        print("="*60)
        print(f"Malha: {self.N} intervalos, h = {self.h:.4f} cm")
        print(f"Tipo: {'Homogêneo' if self.is_homogeneo else 'Heterogêneo'}")
        print(f"Método: {self.metodo_executado}")
        print(f"Precisao comum: {self.precisao}")
        print(f"Dispositivo: {self.nome_dispositivo}")
        if self.k_ref:
            print(f"k_ref: {self.k_ref:.8f} ({self.fonte_ref})")
        print("="*60)
        
        tempo_inicio_solver = time.perf_counter()
        
        for iteracao in range(1, self.max_iter + 1):
            inicio_iter = time.perf_counter()
            inicio_preparacao = time.perf_counter()
            transf_antes = self.tempos_detalhados['tempo_transferencia_numpy_torch_s']
            S_tensor = self.fonte_fissao(self._tensor(phi)) / k_eff
            self._sincronizar()
            transf_preparacao = self.tempos_detalhados['tempo_transferencia_numpy_torch_s'] - transf_antes
            self.tempos_detalhados['tempo_preparacao_fonte_s'] += max(
                0., time.perf_counter()-inicio_preparacao-transf_preparacao)

            inicio_fonte = time.perf_counter()
            transf_antes = (self.tempos_detalhados['tempo_transferencia_numpy_torch_s']
                           + self.tempos_detalhados['tempo_transferencia_torch_numpy_s'])
            phi_novo_tensor = self.resolver_fonte_fixa(S_tensor, chute=phi, metodo=self.metodo_fonte)
            self._sincronizar()
            transf_dentro = (self.tempos_detalhados['tempo_transferencia_numpy_torch_s']
                            + self.tempos_detalhados['tempo_transferencia_torch_numpy_s'] - transf_antes)
            self.tempos_detalhados["tempo_fonte_fixa_s"] += max(0., time.perf_counter()-inicio_fonte-transf_dentro)
            if self.convergiu_fonte_fixa and not self.convergiu_fonte_fixa[-1]:
                self.iteracoes_totais = iteracao
                self.convergiu = False
                self.tempos_iteracao.append(time.perf_counter()-inicio_iter)
                print(
                    "\nFONTE FIXA NÃO CONVERGIU: "
                    f"atingiu max_iter_fonte={self.max_iter_fonte} na iteração externa {iteracao}. "
                    "Ajuste tol_fonte, max_fonte, omega ou amortecimento antes de continuar."
                )
                break
            phi_novo = self._numpy(phi_novo_tensor)
            
            inicio_atualizacao = time.perf_counter()
            integral_novo = self.integral_fissao(phi_novo)
            integral_antigo = self.integral_fissao(phi)
            if abs(integral_antigo) < 1e-30:
                raise ValueError("Integral da fonte muito pequena")
            k_eff_novo = k_eff * (integral_novo / integral_antigo)
            
            max_phi = np.max(phi_novo)
            if max_phi > 0:
                phi_novo = phi_novo / max_phi
            
            erro_k = abs(k_eff_novo - k_eff) / abs(k_eff_novo) if abs(k_eff_novo) > 0 else 1.0
            erro_phi = np.max(np.abs(phi_novo - phi)) / (np.max(np.abs(phi_novo)) + 1e-15)
            self.erro_phi_iterativo = float(erro_phi)
            self.tempos_detalhados["tempo_atualizacao_potencia_s"] += time.perf_counter() - inicio_atualizacao
            
            self.historico_k.append(k_eff_novo)
            if iteracao % 10 == 0 or iteracao == 1:
                self.historico_phi.append(phi_novo.copy())
            self.tempos_iteracao.append(time.perf_counter() - inicio_iter)
            
            phi = phi_novo
            k_eff = k_eff_novo
            
            if self.progress_callback:
                self.progress_callback(iteracao, self.max_iter, k_eff, erro_k, erro_phi)
            
            if iteracao % 10 == 0 or iteracao == 1:
                status_fonte = "ok" if self.convergiu_fonte_fixa[-1] else "max_iter_fonte"
                print(f"Iteração {iteracao:4d}: k_eff = {k_eff:.8f}, "
                      f"erro_k = {erro_k:.2e}, erro_phi = {erro_phi:.2e},"
                      f"Iterações phi: {self.iteracoes_fonte_fixa[-1]}, fonte: {status_fonte}")

            if erro_k < self.tol_k and erro_phi < self.tol_phi:
                self.convergiu = True
                self.iteracoes_totais = iteracao
                print(f"\nCONVERGÊNCIA ALCANÇADA em {iteracao} iterações!")
                print(f"k_eff = {k_eff:.8f}")
                break
        
        if not self.convergiu:
            if self.iteracoes_totais <= 0:
                self.iteracoes_totais = self.max_iter
            print(f"\nNão convergiu após {self.iteracoes_totais} iterações")
        
        self.tempos_detalhados["tempo_loop_iterativo_s"] = time.perf_counter() - tempo_inicio_solver
        self.tempos_detalhados["tempo_montagem_total_s"] = (
            self.tempos_detalhados["tempo_montagem_operador_s"]
            + self.tempos_detalhados["tempo_montagem_hierarquia_s"]
        )
        self.tempos_detalhados["tempo_transferencia_total_s"] = (
            self.tempos_detalhados["tempo_transferencia_numpy_torch_s"]
            + self.tempos_detalhados["tempo_transferencia_torch_numpy_s"]
        )
        self.tempos_detalhados["tempo_iteracao_numerica_s"] = max(
            0.0,
            self.tempos_detalhados['tempo_preparacao_fonte_s']
            + self.tempos_detalhados['tempo_fonte_fixa_s']
            + self.tempos_detalhados['tempo_atualizacao_potencia_s'],
        )
        self.tempo_total = time.perf_counter() - tempo_total_inicio_global
        self.tempo_medio_iteracao = (
            self.tempos_detalhados['tempo_iteracao_numerica_s'] / max(self.iteracoes_totais, 1)
            if self.iteracoes_totais else 0.0
        )
        self.tempos_detalhados["tempo_total_s"] = self.tempo_total
        self.tempos_detalhados["tempo_medio_iteracao_s"] = self.tempo_medio_iteracao
        self.k_eff = k_eff
        self.phi = phi
        
        if self.k_ref:
            self.erro_k = abs(self.k_eff - self.k_ref) / abs(self.k_ref)
        
        if self.phi_ref is not None:
            for ponto, val_ref in zip(self.pontos_interesse, self.fluxo_analitico_pontos(self.pontos_interesse)):
                val_calc = np.interp(ponto, self.x, self.phi)
                self.erro_phi_pontos[ponto] = (abs(val_calc-val_ref)/abs(val_ref)
                                              if abs(val_ref) > 1e-12 else None)
        
        erros_validos = [v for v in self.erro_phi_pontos.values() if v is not None]
        self.erro_phi_max = (
            float(np.max(erros_validos))
            if erros_validos else 0.0
        )
        self.phi_normalizado = self.normalizar_potencia(phi)
        self.tempo_total = time.perf_counter() - tempo_total_inicio_global
        self.tempos_detalhados['tempo_total_s'] = self.tempo_total
        return k_eff, self.phi_normalizado

    def resumo_resultado(self, caso=None):
        return {
            "Caso": caso or ("Homogêneo" if self.is_homogeneo else "Heterogêneo"),
            "Discretizacao": self.discretizacao.upper(),
            "Nos": self.N + 1,
            "Precisao": str(self.dtype),
            "Ciclo": "V" if self.solver_fonte_fixa is not None else "nao se aplica",
            "Niveis": len(self.solver_fonte_fixa.operadores) if self.solver_fonte_fixa is not None else 0,
            "Pre_suavizacoes": self.solver_fonte_fixa.pre_suavizacoes if self.solver_fonte_fixa is not None else 0,
            "Post_suavizacoes": self.solver_fonte_fixa.post_suavizacoes if self.solver_fonte_fixa is not None else 0,
            "Suavizacoes_grosso": self.solver_fonte_fixa.suavizacoes_grosso if self.solver_fonte_fixa is not None else 0,
            "Método": self.nome_metodo(self.metodo_fonte),
            "N": self.N,
            "k_eff": self.k_eff,
            "Referência": self.k_ref,
            "Erro k (%)": self.erro_k * 100.0 if self.erro_k is not None else None,
            "Iter. externas": self.iteracoes_totais,
            "Iter. fonte média": float(np.mean(self.iteracoes_fonte_fixa)) if self.iteracoes_fonte_fixa else 0.0,
            "Resíduo final": float(self.residuos_fonte_fixa[-1]) if self.residuos_fonte_fixa else None,
            "Fonte fixa convergiu (todas as chamadas)": bool(self.convergiu_fonte_fixa) and all(self.convergiu_fonte_fixa),
            "Chamadas fonte fixa não convergidas": int(sum(1 for ok in self.convergiu_fonte_fixa if not ok)),
            "Bateu max_iter externo": not self.convergiu,
            "Tempo (s)": self.tempo_total,
            "tempo_montagem_operador_s": self.tempos_detalhados.get("tempo_montagem_operador_s", 0.0),
            "tempo_montagem_hierarquia_s": self.tempos_detalhados.get("tempo_montagem_hierarquia_s", 0.0),
            "tempo_montagem_total_s": self.tempos_detalhados.get("tempo_montagem_total_s", 0.0),
            "tempo_transferencia_numpy_torch_s": self.tempos_detalhados.get("tempo_transferencia_numpy_torch_s", 0.0),
            "tempo_transferencia_torch_numpy_s": self.tempos_detalhados.get("tempo_transferencia_torch_numpy_s", 0.0),
            "tempo_transferencia_total_s": self.tempos_detalhados.get("tempo_transferencia_total_s", 0.0),
            "tempo_loop_iterativo_s": self.tempos_detalhados.get("tempo_loop_iterativo_s", 0.0),
            "tempo_iteracao_numerica_s": self.tempos_detalhados.get("tempo_iteracao_numerica_s", 0.0),
            "tempo_fonte_fixa_s": self.tempos_detalhados.get("tempo_fonte_fixa_s", 0.0),
            "tempo_atualizacao_potencia_s": self.tempos_detalhados.get("tempo_atualizacao_potencia_s", 0.0),
            "tempo_medio_iteracao_s": self.tempos_detalhados.get("tempo_medio_iteracao_s", self.tempo_medio_iteracao),
            "tol_fonte": self.tol_fonte,
            "max_iter_fonte": self.max_iter_fonte,
            "omega_fonte": self.omega_fonte,
            "amortecimento_unet": self.amortecimento_unet,
        }


def executar_comparacao_resolvedores(config_base, metodos=("unet_multigrid", "thomas")):
    resultados = []
    solvers = []
    for metodo in metodos:
        cfg = dict(config_base)
        cfg["metodo_fonte"] = metodo
        if metodo == "thomas":
            cfg["permitir_thomas_comparacao"] = True
        solver = SolverDifusaoAI4PDEs(**cfg)
        solver.resolver()
        resultados.append(solver.resumo_resultado())
        solvers.append(solver)
    return resultados, solvers
