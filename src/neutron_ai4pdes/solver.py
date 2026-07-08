"""Backend numérico para difusão de nêutrons 1D."""

import platform
import time

import numpy as np
import torch

from .models import OperadorDifusao1D, SolverFonteFixaUNet1D
from .references import calcular_k_eff_analitico, fluxo_analitico_homogeneo


class SolverDifusaoAI4PDEs:
    REFERENCIAS = {
        'homogeneo': {'k_eff': None, 'fonte': 'Analítica'},
        'heterogeneo': {'k_eff': 1.09506, 'fonte': 'Nozimar'}
    }
    
    def __init__(self, L, N, materiais, cond_esquerda='reflexiva', cond_direita='vácuo',
                 tol_k=1e-6, tol_phi=1e-5, max_iter=1000, potencia_nominal=100.0,
                 pontos_interesse=None, progress_callback=None, dispositivo_preferido='auto',
                 omega_fonte=0.75, amortecimento_unet=0.20,
                 tol_fonte=None, max_iter_fonte=5000):
        
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
            "AI4PDEs 1D adaptado com PyTorch CUDA"
            if self.device.type == 'cuda'
            else "AI4PDEs 1D adaptado com PyTorch CPU"
        )
        self.modelo_A = None
        self.solver_fonte_fixa = None
        self.iteracoes_fonte_fixa = []
        self.residuos_fonte_fixa = []
        
        self.k_eff = None
        self.phi = None
        self.phi_normalizado = None
        self.historico_k = []
        self.historico_phi = []
        self.tempos_iteracao = []
        self.tempo_total = 0.0
        self.tempo_medio_iteracao = 0.0
        
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

    def obter_nome_dispositivo(self):
        if self.device.type == 'cuda':
            try:
                return torch.cuda.get_device_name(0)
            except Exception:
                return "GPU CUDA"
        return platform.processor() or platform.machine() or "CPU"
    
    def criar_operador_A(self):
        self.modelo_A = OperadorDifusao1D(
            self.D_arr,
            self.Sigma_a_arr,
            self.h,
            self.cond_esquerda,
            self.cond_direita,
        ).to(self.device)
        self.solver_fonte_fixa = SolverFonteFixaUNet1D(
            self.modelo_A,
            omega=self.omega_fonte,
            amortecimento_unet=self.amortecimento_unet,
        ).to(self.device)
        self.metodo_executado = (
            "AI4PDEs 1D adaptado - stencil/convolução fixa + multiescala"
        )
    
    def resolver_sistema(self, S, chute=None):
        if self.solver_fonte_fixa is None:
            raise RuntimeError("O solver AI4PDEs 1D ainda não foi criado.")
        S_tensor = S.to(self.device).float() if torch.is_tensor(S) else torch.tensor(S, device=self.device, dtype=torch.float32)
        chute_tensor = None
        if chute is not None:
            chute_tensor = chute.to(self.device).float() if torch.is_tensor(chute) else torch.tensor(chute, device=self.device, dtype=torch.float32)
        phi, n_iter, residuo = self.solver_fonte_fixa.resolver(
            S_tensor,
            chute=chute_tensor,
            tol=self.tol_fonte,
            max_iter=self.max_iter_fonte,
        )
        self.iteracoes_fonte_fixa.append(int(n_iter))
        self.residuos_fonte_fixa.append(float(residuo))
        return phi
    
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
        P_atual = 3.2e-11 * np.sum(self.nuSigma_f_arr * phi * self.h)
        if P_atual > 0:
            return phi * (self.potencia_nominal / P_atual)
        return phi
    
    def resolver(self):
        self.criar_operador_A()
        self.calcular_referencias()
        n = self.N + 1
        nuSigma_f = self.nuSigma_f_arr
        phi = np.ones(n)
        k_eff = 1.0
        self.historico_k = []
        self.historico_phi = []
        self.tempos_iteracao = []
        self.iteracoes_fonte_fixa = []
        self.residuos_fonte_fixa = []
        
        print("\n" + "="*60)
        print("SOLVER DE DIFUSÃO DE NÊUTRONS COM AI4PDEs 1D ADAPTADO")
        print("="*60)
        print(f"Malha: {self.N} intervalos, h = {self.h:.4f} cm")
        print(f"Tipo: {'Homogêneo' if self.is_homogeneo else 'Heterogêneo'}")
        print(f"Método: {self.metodo_executado}")
        print(f"Dispositivo: {self.nome_dispositivo}")
        if self.k_ref:
            print(f"k_ref: {self.k_ref:.8f} ({self.fonte_ref})")
        print("="*60)
        
        tempo_inicio_solver = time.perf_counter()
        
        for iteracao in range(1, self.max_iter + 1):
            inicio_iter = time.perf_counter()
            S = (1.0 / k_eff) * nuSigma_f * phi
            S_tensor = torch.tensor(S, device=self.device, dtype=torch.float32)
            phi_novo_tensor = self.resolver_sistema(S_tensor, chute=phi)
            phi_novo = phi_novo_tensor.cpu().numpy()
            
            integral_novo = self.calcular_integral(nuSigma_f * phi_novo)
            integral_antigo = self.calcular_integral(nuSigma_f * phi)
            if abs(integral_antigo) < 1e-30:
                raise ValueError("Integral da fonte muito pequena")
            k_eff_novo = k_eff * (integral_novo / integral_antigo)
            
            max_phi = np.max(phi_novo)
            if max_phi > 0:
                phi_novo = phi_novo / max_phi
            
            erro_k = abs(k_eff_novo - k_eff) / abs(k_eff_novo) if abs(k_eff_novo) > 0 else 1.0
            erro_phi = np.max(np.abs(phi_novo - phi)) / (np.max(np.abs(phi_novo)) + 1e-15)
            self.erro_phi_iterativo = float(erro_phi)
            
            self.historico_k.append(k_eff_novo)
            if iteracao % 10 == 0 or iteracao == 1:
                self.historico_phi.append(phi_novo.copy())
            self.tempos_iteracao.append(time.perf_counter() - inicio_iter)
            
            phi = phi_novo
            k_eff = k_eff_novo
            
            if self.progress_callback:
                self.progress_callback(iteracao, self.max_iter, k_eff, erro_k, erro_phi)
            
            if iteracao % 10 == 0 or iteracao == 1:
                print(f"Iteração {iteracao:4d}: k_eff = {k_eff:.8f}, "
                      f"erro_k = {erro_k:.2e}, erro_phi = {erro_phi:.2e},"
                      f"Iterações phi: {self.iteracoes_fonte_fixa[-1]}")

            if erro_k < self.tol_k and erro_phi < self.tol_phi:
                self.convergiu = True
                self.iteracoes_totais = iteracao
                print(f"\n✅ CONVERGÊNCIA ALCANÇADA em {iteracao} iterações!")
                print(f"k_eff = {k_eff:.8f}")
                break
        
        if not self.convergiu:
            self.iteracoes_totais = self.max_iter
            print(f"\n⚠️ Não convergiu após {self.max_iter} iterações")
        
        self.tempo_total = time.perf_counter() - tempo_inicio_solver
        self.tempo_medio_iteracao = (
            self.tempo_total / max(self.iteracoes_totais, 1)
            if self.iteracoes_totais else 0.0
        )
        self.k_eff = k_eff
        self.phi = phi
        
        if self.k_ref:
            self.erro_k = abs(self.k_eff - self.k_ref) / abs(self.k_ref)
        
        if self.phi_ref is not None:
            for ponto in self.pontos_interesse:
                idx = np.argmin(np.abs(self.x - ponto))
                if idx < len(self.phi) and idx < len(self.phi_ref):
                    val_calc = self.phi[idx]
                    val_ref = self.phi_ref[idx]
                    if abs(val_ref) > 1e-15:
                        self.erro_phi_pontos[ponto] = abs(val_calc - val_ref) / abs(val_ref)
                    else:
                        self.erro_phi_pontos[ponto] = abs(val_calc - val_ref)
        
        self.erro_phi_max = (
            float(np.max(list(self.erro_phi_pontos.values())))
            if self.erro_phi_pontos else 0.0
        )
        self.phi_normalizado = self.normalizar_potencia(phi)
        return k_eff, self.phi_normalizado


# ============================================================================
# 4. INTERFACE GRÁFICA
# ============================================================================
