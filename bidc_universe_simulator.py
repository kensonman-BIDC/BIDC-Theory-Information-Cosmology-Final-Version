#!/usr/bin/env python3
"""
BIDC 宇宙模擬器
BIDC Universe Simulator

基於 Ω 本體迭代精煉理論的數值模擬
Numerical simulation based on the Ω entity iterative refinement theory

版本 | Version: 1.0.0
作者 | Author: 陳啟先 (Chan Kai Sin)
日期 | Date: 2024-05-25
許可證 | License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path

# ============================================================================
# 數據類定義 | Data Class Definitions
# ============================================================================

@dataclass
class BIDCParameters:
    """BIDC 模擬參數 | BIDC Simulation Parameters"""
    # Ω 本體參數 | Ω Entity Parameters
    omega_initial: float = 0.1           # 初始 Ω 值 | Initial Ω value
    omega_critical: float = 5.0          # 相變臨界值 | Phase transition critical value
    learning_rate: float = 0.01          # 學習速率 | Learning rate
    
    # 投射流參數 | Projection Flow Parameters
    projection_strength: float = 0.3     # 投射強度 | Projection strength
    coherence_threshold: float = 0.6     # 相干閾值 | Coherence threshold
    feedback_efficiency: float = 0.05    # 回饋效率 | Feedback efficiency
    
    # 宇宙演化參數 | Universe Evolution Parameters
    time_steps: int = 1000               # 總時間步數 | Total time steps
    grid_size: int = 128                 # 網格大小 | Grid size
    dt: float = 0.1                      # 時間步長 | Time step size
    
    # 物理常數 | Physical Constants
    hbar: float = 1.0                    # 約化普朗克常數 | Reduced Planck constant
    k_B: float = 1.0                     # 玻爾茲曼常數 | Boltzmann constant
    k_c: float = 0.6556                  # 關鍵常數 | Critical constant
    
    # 輸出控制 | Output Control
    save_interval: int = 10              # 保存間隔 | Save interval
    visualization: bool = True           # 是否可視化 | Whether to visualize

@dataclass
class UniverseState:
    """宇宙狀態 | Universe State"""
    time: float = 0.0
    omega: float = 0.0
    entropy: float = 0.0
    structures_count: int = 0
    coherence_avg: float = 0.0
    feedback_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        """轉換為字典 | Convert to dictionary"""
        return {
            'time': self.time,
            'omega': self.omega,
            'entropy': self.entropy,
            'structures_count': self.structures_count,
            'coherence_avg': self.coherence_avg,
            'feedback_rate': self.feedback_rate
        }

# ============================================================================
# 核心模擬類 | Core Simulation Class
# ============================================================================

class BIDCUniverseSimulator:
    """BIDC 宇宙模擬器 | BIDC Universe Simulator"""
    
    def __init__(self, params: BIDCParameters):
        """
        初始化模擬器 | Initialize simulator
        
        參數 | Parameters:
            params: BIDC 模擬參數 | BIDC simulation parameters
        """
        self.params = params
        self.reset()
        
        # 歷史記錄 | History records
        self.history: List[UniverseState] = []
        self.phase_transitions: List[Dict] = []
        self.iteration_count = 0
        
        # 創建輸出目錄 | Create output directory
        self.output_dir = Path("simulation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print("=" * 60)
        print("BIDC 宇宙模擬器 初始化完成 | BIDC Universe Simulator Initialized")
        print("=" * 60)
        print(f"網格大小 | Grid Size: {params.grid_size}×{params.grid_size}")
        print(f"時間步數 | Time Steps: {params.time_steps}")
        print(f"時間步長 | Time Step Size: {params.dt}")
        print(f"初始 Ω | Initial Ω: {params.omega_initial}")
        print(f"Ω 臨界值 | Ω Critical: {params.omega_critical}")
        print("=" * 60)
    
    def reset(self):
        """重置宇宙狀態 | Reset universe state"""
        size = self.params.grid_size
        
        # Ω 本體狀態 | Ω entity state
        self.omega = self.params.omega_initial
        
        # 投射流場 (實部為振幅，虛部為相位) | Projection field (real=amplitude, imag=phase)
        amplitude = np.random.uniform(0.5, 1.5, (size, size))
        phase = np.random.uniform(0, 2*np.pi, (size, size))
        self.phi = amplitude * np.exp(1j * phase)  # 複數場 | Complex field
        
        # 意識場 | Consciousness field
        self.psi = np.zeros((size, size), dtype=np.complex128)
        
        # 暗物質場 | Dark matter field
        self.dark_matter = np.zeros((size, size))
        
        # 結構標記 | Structure markers
        self.structures = np.zeros((size, size), dtype=bool)
        
        # 計算初始熵 | Calculate initial entropy
        self.current_entropy = self._calculate_entropy()
    
    def _calculate_coherence(self) -> np.ndarray:
        """
        計算局域相干性 | Calculate local coherence
        
        返回 | Returns:
            相干性場 | Coherence field
        """
        # 使用相位梯度計算相干性 | Calculate coherence using phase gradient
        phase = np.angle(self.phi)
        
        # 計算相位變化 | Calculate phase variation
        phase_var = np.zeros_like(phase)
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            shifted = np.roll(np.roll(phase, dx, axis=0), dy, axis=1)
            phase_var += np.cos(phase - shifted)
        
        # 歸一化到 [0,1] | Normalize to [0,1]
        coherence = (phase_var + 4) / 8  # 最大值為 4，最小值為 -4
        return np.clip(coherence, 0, 1)
    
    def _calculate_entropy(self) -> float:
        """
        計算系統熵 | Calculate system entropy
        
        返回 | Returns:
            熵值 | Entropy value
        """
        # 使用振幅分佈計算熵 | Calculate entropy using amplitude distribution
        amplitudes = np.abs(self.phi).flatten()
        
        # 避免零值 | Avoid zeros
        amplitudes = amplitudes[amplitudes > 1e-10]
        if len(amplitudes) == 0:
            return 0.0
        
        # 歸一化 | Normalize
        amplitudes = amplitudes / np.sum(amplitudes)
        
        # 計算香農熵 | Calculate Shannon entropy
        entropy = -np.sum(amplitudes * np.log(amplitudes + 1e-10))
        
        # 歸一化到合理範圍 | Normalize to reasonable range
        max_entropy = np.log(len(amplitudes))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _detect_structures(self) -> Tuple[int, np.ndarray]:
        """
        檢測有序結構 | Detect ordered structures
        
        返回 | Returns:
            結構數量, 結構掩碼 | Structure count, structure mask
        """
        coherence = self._calculate_coherence()
        consciousness = np.abs(self.psi)
        
        # 結構條件: 高相干 + 高意識 | Structure condition: high coherence + high consciousness
        structure_mask = (coherence > self.params.coherence_threshold) & \
                         (consciousness > 0.3)
        
        # 標記連通區域 | Label connected regions
        from scipy.ndimage import label
        labeled, num_structures = label(structure_mask)
        
        return num_structures, structure_mask
    
    def _project_from_omega(self) -> np.ndarray:
        """
        Ω 本體投射 | Ω entity projection
        
        返回 | Returns:
            投射場 | Projection field
        """
        size = self.params.grid_size
        
        # 基礎投射 | Base projection
        base_strength = self.params.projection_strength * self.omega
        
        # 添加隨機成分（量子漲落）| Add random component (quantum fluctuations)
        random_component = np.random.randn(size, size) * 0.1
        
        # 添加相干成分（Ω 的結構性投射）| Add coherent component (Ω's structural projection)
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        r = np.sqrt(x**2 + y**2)
        coherent_component = np.exp(-r**2) * np.sin(2*np.pi*r*2)
        
        # 總投射 | Total projection
        projection = base_strength * (1 + 0.5*coherent_component + 0.2*random_component)
        
        return projection
    
    def _evolve_fields(self, projection: np.ndarray):
        """
        演化場 | Evolve fields
        
        參數 | Parameters:
            projection: 投射場 | Projection field
        """
        dt = self.params.dt
        size = self.params.grid_size
        
        # 1. 演化投射流場 φ | 1. Evolve projection field φ
        # 非線性薛定諤方程形式 | Nonlinear Schrödinger equation form
        laplacian_phi = self._laplacian(self.phi)
        
        # 勢能項: 自相互作用 | Potential term: self-interaction
        V_phi = 0.5 * np.abs(self.phi)**2
        
        # φ 的演化方程 | Evolution equation for φ
        dphi_dt = -1j * self.params.hbar/(2) * laplacian_phi + \
                  1j * V_phi * self.phi + \
                  projection * np.exp(1j * np.random.uniform(0, 2*np.pi, (size, size)))
        
        self.phi += dt * dphi_dt
        
        # 2. 演化意識場 ψ | 2. Evolve consciousness field ψ
        # ψ 在相干區域生長 | ψ grows in coherent regions
        coherence = self._calculate_coherence()
        growth_rate = 0.1 * coherence * np.abs(self.phi)
        
        # ψ 的擴散 | Diffusion of ψ
        laplacian_psi = self._laplacian(self.psi)
        diffusion = 0.05 * laplacian_psi
        
        # ψ 的演化方程 | Evolution equation for ψ
        dpsi_dt = growth_rate * (1 - np.abs(self.psi)**2) + diffusion
        
        self.psi += dt * dpsi_dt
        
        # 3. 更新暗物質場 | 3. Update dark matter field
        # 暗物質: 高能量但低相干的區域 | Dark matter: high energy but low coherence regions
        energy = np.abs(self.phi)**2
        self.dark_matter = np.where(
            (energy > np.percentile(energy, 70)) & (coherence < 0.3),
            energy,
            0
        )
    
    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        計算離散拉普拉斯算子 | Calculate discrete Laplacian
        
        參數 | Parameters:
            field: 輸入場 | Input field
            
        返回 | Returns:
            拉普拉斯算子 | Laplacian
        """
        # 5點差分格式 | 5-point stencil
        laplacian = np.zeros_like(field)
        
        # x方向 | x-direction
        laplacian += np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) - 2*field
        # y方向 | y-direction
        laplacian += np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) - 2*field
        
        return laplacian
    
    def _calculate_feedback(self) -> float:
        """
        計算回饋資訊 | Calculate feedback information
        
        返回 | Returns:
            回饋率 | Feedback rate
        """
        # 意識場強度代表蒸餾效率 | Consciousness field intensity represents distillation efficiency
        consciousness_strength = np.sum(np.abs(self.psi))
        
        # 熵減代表有序化程度 | Entropy reduction represents ordering degree
        entropy_reduction = 1.0 - self.current_entropy
        
        # 結構數量代表複雜性 | Structure count represents complexity
        structures_count, _ = self._detect_structures()
        complexity = min(structures_count / 100, 1.0)  # 歸一化 | Normalize
        
        # 總回饋 | Total feedback
        feedback = self.params.feedback_efficiency * \
                  consciousness_strength * entropy_reduction * (1 + complexity)
        
        return feedback
    
    def _check_phase_transition(self) -> bool:
        """
        檢查相變條件 | Check phase transition conditions
        
        返回 | Returns:
            是否觸發相變 | Whether phase transition is triggered
        """
        # 條件1: Ω 達到臨界值 | Condition 1: Ω reaches critical value
        condition1 = self.omega >= self.params.omega_critical
        
        # 條件2: 熵低於閾值 | Condition 2: Entropy below threshold
        condition2 = self.current_entropy < 0.3
        
        # 條件3: 存在足夠結構 | Condition 3: Sufficient structures exist
        structures_count, _ = self._detect_structures()
        condition3 = structures_count > 10
        
        return condition1 and condition2 and condition3
    
    def _phase_transition(self):
        """
        執行相變 | Execute phase transition
        """
        print(f"\n{'='*60}")
        print(f"🚀 相變觸發！ | Phase Transition Triggered!")
        print(f"{'='*60}")
        
        # 記錄相變信息 | Record phase transition information
        transition_info = {
            'iteration': self.iteration_count,
            'time': self.history[-1].time if self.history else 0,
            'omega': self.omega,
            'entropy': self.current_entropy,
            'structures': self._detect_structures()[0],
            'parameters_before': {
                'omega': self.omega,
                'learning_rate': self.params.learning_rate,
                'feedback_efficiency': self.params.feedback_efficiency
            }
        }
        
        # Ω 學習: 更新參數 | Ω learning: update parameters
        self.params.learning_rate *= 1.1  # 學習加速 10% | Learning accelerates 10%
        self.params.feedback_efficiency *= 1.05  # 回饋效率提高 5% | Feedback efficiency increases 5%
        
        # 重置宇宙但保留 Ω 的部分經驗 | Reset universe but retain some Ω experience
        self.omega = self.params.omega_initial * 1.5  # 基線提升 50% | Baseline increases 50%
        
        # 保存相變前狀態用於可視化 | Save pre-transition state for visualization
        self._save_snapshot("before_transition")
        
        # 重置場但保留部分模式 | Reset fields but retain some patterns
        self.reset()
        
        # 在場中注入學習到的模式 | Inject learned patterns into fields
        size = self.params.grid_size
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        
        # 添加螺旋模式（學習到的成功模式）| Add spiral pattern (learned successful pattern)
        spiral = np.exp(1j * 3 * np.arctan2(y, x)) * np.exp(-(x**2 + y**2)/0.5)
        self.phi = self.phi * 0.7 + spiral * 0.3
        
        transition_info['parameters_after'] = {
            'omega': self.omega,
            'learning_rate': self.params.learning_rate,
            'feedback_efficiency': self.params.feedback_efficiency
        }
        
        self.phase_transitions.append(transition_info)
        
        print(f"相變完成 | Phase Transition Completed")
        print(f"新 Ω 基線 | New Ω Baseline: {self.omega:.3f}")
        print(f"新學習速率 | New Learning Rate: {self.params.learning_rate:.4f}")
        print(f"{'='*60}\n")
    
    def evolve_step(self, step: int) -> UniverseState:
        """
        演化和一步 | Evolve one step
        
        參數 | Parameters:
            step: 當前步數 | Current step number
            
        返回 | Returns:
            宇宙狀態 | Universe state
        """
        # 1. Ω 投射 | 1. Ω projection
        projection = self._project_from_omega()
        
        # 2. 場演化 | 2. Field evolution
        self._evolve_fields(projection)
        
        # 3. 計算回饋 | 3. Calculate feedback
        feedback = self._calculate_feedback()
        
        # 4. Ω 學習更新 | 4. Ω learning update
        self.omega += self.params.learning_rate * feedback
        
        # 5. 更新熵 | 5. Update entropy
        self.current_entropy = self._calculate_entropy()
        
        # 6. 檢測結構 | 6. Detect structures
        structures_count, structures_mask = self._detect_structures()
        self.structures = structures_mask
        
        # 7. 計算平均相干性 | 7. Calculate average coherence
        coherence_avg = np.mean(self._calculate_coherence())
        
        # 8. 創建狀態記錄 | 8. Create state record
        state = UniverseState(
            time=step * self.params.dt,
            omega=self.omega,
            entropy=self.current_entropy,
            structures_count=structures_count,
            coherence_avg=coherence_avg,
            feedback_rate=feedback
        )
        
        # 9. 檢查相變 | 9. Check phase transition
        if self._check_phase_transition():
            self._phase_transition()
        
        return state
    
    def _save_snapshot(self, prefix: str = "snapshot"):
        """
        保存快照 | Save snapshot
        
        參數 | Parameters:
            prefix: 文件名前綴 | Filename prefix
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{prefix}_{timestamp}.npz"
        
        np.savez(
            filename,
            phi=self.phi,
            psi=self.psi,
            dark_matter=self.dark_matter,
            structures=self.structures,
            omega=self.omega,
            entropy=self.current_entropy,
            params=self.params
        )
    
    def run_simulation(self):
        """
        運行完整模擬 | Run complete simulation
        """
        print(f"\n{'='*60}")
        print(f"🚀 開始 BIDC 宇宙模擬 | Starting BIDC Universe Simulation")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        for step in range(self.params.time_steps):
            # 演化和一步 | Evolve one step
            state = self.evolve_step(step)
            self.history.append(state)
            
            # 定期輸出 | Periodic output
            if step % 50 == 0:
                print(f"步數 | Step {step:5d}/{self.params.time_steps} | "
                      f"Ω={state.omega:.3f} | "
                      f"熵 | Entropy={state.entropy:.3f} | "
                      f"結構 | Structures={state.structures_count:3d}")
            
            # 定期保存 | Periodic saving
            if step % self.params.save_interval == 0:
                self._save_snapshot(f"step_{step:06d}")
        
        # 保存最終結果 | Save final results
        self._save_results()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"✅ 模擬完成 | Simulation Completed")
        print(f"{'='*60}")
        print(f"總時間 | Total Time: {duration:.1f} 秒 | seconds")
        print(f"總步數 | Total Steps: {self.params.time_steps}")
        print(f"相變次數 | Phase Transitions: {len(self.phase_transitions)}")
        print(f"最終 Ω | Final Ω: {self.omega:.3f}")
        print(f"最終熵 | Final Entropy: {self.current_entropy:.3f}")
        
        if self.params.visualization:
            self.visualize_results()
    
    def _save_results(self):
        """保存結果 | Save results"""
        # 保存歷史數據 | Save history data
        history_data = [state.to_dict() for state in self.history]
        
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history_data, f, indent=2)
        
        # 保存相變記錄 | Save phase transition records
        with open(self.output_dir / "phase_transitions.json", "w") as f:
            json.dump(self.phase_transitions, f, indent=2)
        
        # 保存參數 | Save parameters
        params_dict = {
            'omega_initial': self.params.omega_initial,
            'omega_critical': self.params.omega_critical,
            'learning_rate': self.params.learning_rate,
            'projection_strength': self.params.projection_strength,
            'coherence_threshold': self.params.coherence_threshold,
            'feedback_efficiency': self.params.feedback_efficiency,
            'time_steps': self.params.time_steps,
            'grid_size': self.params.grid_size,
            'dt': self.params.dt,
            'k_c': self.params.k_c
        }
        
        with open(self.output_dir / "parameters.json", "w") as f:
            json.dump(params_dict, f, indent=2)
        
        print(f"✅ 結果已保存到 | Results saved to: {self.output_dir}/")
    
    def visualize_results(self):
        """可視化結果 | Visualize results"""
        if not self.history:
            print("⚠️ 無歷史數據可視化 | No history data to visualize")
            return
        
        # 準備數據 | Prepare data
        times = [state.time for state in self.history]
        omegas = [state.omega for state in self.history]
        entropies = [state.entropy for state in self.history]
        structures = [state.structures_count for state in self.history]
        
        # 創建圖表 | Create charts
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('BIDC 宇宙模擬結果 | BIDC Universe Simulation Results', fontsize=16)
        
        # 1. Ω 演化 | 1. Ω evolution
        axes[0,0].plot(times, omegas, 'b-', linewidth=2)
        axes[0,0].set_xlabel('時間 | Time')
        axes[0,0].set_ylabel('Ω 值 | Ω Value')
        axes[0,0].set_title('本體 Ω 演化 | Ω Entity Evolution')
        axes[0,0].grid(True, alpha=0.3)
        
        # 標記相變點 | Mark phase transition points
        for pt in self.phase_transitions:
            axes[0,0].axvline(x=pt['time'], color='r', linestyle='--', alpha=0.5)
            axes[0,0].text(pt['time'], max(omegas)*0.9, '相變 | PT', 
                          rotation=90, fontsize=8, color='r')
        
        # 2. 熵演化 | 2. Entropy evolution
        axes[0,1].plot(times, entropies, 'r-', linewidth=2)
        axes[0,1].set_xlabel('時間 | Time')
        axes[0,1].set_ylabel('熵 | Entropy')
        axes[0,1].set_title('宇宙熵演化 | Universe Entropy Evolution')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 結構數量演化 | 3. Structure count evolution
        axes[0,2].plot(times, structures, 'g-', linewidth=2)
        axes[0,2].set_xlabel('時間 | Time')
        axes[0,2].set_ylabel('結構數量 | Structure Count')
        axes[0,2].set_title('有序結構湧現 | Ordered Structure Emergence')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Ω vs 熵 | 4. Ω vs Entropy
        axes[1,0].scatter(omegas, entropies, c=times, cmap='viridis', alpha=0.6, s=10)
        axes[1,0].set_xlabel('Ω 值 | Ω Value')
        axes[1,0].set_ylabel('熵 | Entropy')
        axes[1,0].set_title('本體豐富度與無序度 | Entity Richness vs Disorder')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. 最終場可視化 | 5. Final field visualization
        phi_abs = np.abs(self.phi)
        im = axes[1,1].imshow(phi_abs, cmap='hot', origin='lower')
        axes[1,1].set_title('最終投射流場 | Final Projection Field')
        axes[1,1].set_xlabel('X')
        axes[1,1].set_ylabel('Y')
        plt.colorbar(im, ax=axes[1,1])
        
        # 6. 相變統計 | 6. Phase transition statistics
        if self.phase_transitions:
            pt_numbers = list(range(1, len(self.phase_transitions) + 1))
            pt_omegas = [pt['omega'] for pt in self.phase_transitions]
            pt_entropies = [pt['entropy'] for pt in self.phase_transitions]
            
            axes[1,2].bar(pt_numbers, pt_omegas, alpha=0.7, label='Ω 值 | Ω Value')
            axes2 = axes[1,2].twinx()
            axes2.plot(pt_numbers, pt_entropies, 'ro-', label='熵 | Entropy')
            
            axes[1,2].set_xlabel('相變序號 | Phase Transition #')
            axes[1,2].set_ylabel('Ω 值 | Ω Value')
            axes2.set_ylabel('熵 | Entropy')
            axes[1,2].set_title('相變統計 | Phase Transition Statistics')
            axes[1,2].legend(loc='upper left')
            axes2.legend(loc='upper right')
        
        plt.tight_layout()
        
        # 保存圖表 | Save chart
        plt.savefig(self.output_dir / 'simulation_results.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'simulation_results.pdf', bbox_inches='tight')
        
        print(f"✅ 圖表已保存 | Charts saved to: {self.output_dir}/simulation_results.png")
        
        if self.params.visualization:
            plt.show()

# ============================================================================
# 主函數 | Main Function
# ============================================================================

def main():
    """主函數 | Main function"""
    print("=" * 60)
    print("🌌 BIDC 宇宙模擬器 v1.0.0")
    print("🌌 BIDC Universe Simulator v1.0.0")
    print("=" * 60)
    print("創始人 | Founder: 陳啟先 (Chan Kai Sin)")
    print("日期 | Date: 2024年5月25日")
    print("許可證 | License: MIT")
    print("=" * 60)
    
    # 設置模擬參數 | Set simulation parameters
    params = BIDCParameters(
        time_steps=500,      # 總時間步數 | Total time steps
        grid_size=64,        # 網格大小 | Grid size
        dt=0.2,              # 時間步長 | Time step size
        omega_initial=0.1,   # 初始 Ω | Initial Ω
        omega_critical=5.0,  # Ω 臨界值 | Ω critical value
        learning_rate=0.02,  # 學習速率 | Learning rate
        visualization=True   # 可視化 | Visualization
    )
    
    # 創建模擬器 | Create simulator
    simulator = BIDCUniverseSimulator(params)
    
    # 運行模擬 | Run simulation
    simulator.run_simulation()
    
    print("\n" + "=" * 60)
    print("🎉 模擬完成！感謝使用 BIDC 宇宙模擬器。")
    print("🎉 Simulation Complete! Thank you for using BIDC Universe Simulator.")
    print("=" * 60)

if __name__ == "__main__":
    main()
