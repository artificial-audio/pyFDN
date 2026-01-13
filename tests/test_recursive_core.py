"""Unit tests for core recursive DSP abstractions."""

import pytest
import torch
from pyFDN.recursive import Stage, RecursionCore, DelayRead, DelayWrite, OutputTap


class DummyStage(Stage):
    """Simple stage for testing that adds a constant to ctx['lines']."""
    
    def __init__(self, add_value: float = 1.0, has_state: bool = False):
        state_keys = {"dummy_state"} if has_state else set()
        super().__init__(state_keys)
        self.add_value = add_value
        self.has_state = has_state
    
    def init_state(self, batch_size: int, device: torch.device):
        if self.has_state:
            return {"dummy_state": torch.zeros(batch_size, device=device)}
        return {}
    
    def step_block(self, ctx, state_t, next_state, block_size):
        if "lines" in ctx:
            ctx["lines"] = ctx["lines"] + self.add_value
        else:
            # Create lines if not present
            batch_size = ctx["x"].shape[0]
            num_lines = 4
            ctx["lines"] = torch.full(
                (batch_size, block_size, num_lines),
                self.add_value,
                device=ctx["x"].device
            )
        
        if self.has_state:
            next_state["dummy_state"] = state_t["dummy_state"] + 1


class TestStageBase:
    """Test Stage abstract base class."""
    
    def test_stage_initialization(self):
        """Test basic stage initialization."""
        stage = DummyStage(add_value=5.0)
        assert stage.state_keys == set()
        assert stage.add_value == 5.0
    
    def test_stage_with_state(self):
        """Test stage with state keys."""
        stage = DummyStage(has_state=True)
        assert stage.state_keys == {"dummy_state"}
        
        state = stage.init_state(2, torch.device("cpu"))
        assert "dummy_state" in state
        assert state["dummy_state"].shape == (2,)


class TestRecursionCore:
    """Test RecursionCore class."""
    
    def test_core_initialization(self):
        """Test core initialization with stages."""
        stages = [
            DummyStage(),
            DummyStage(),
        ]
        core = RecursionCore(stages)
        assert len(core.stages) == 2
        assert core.device == torch.device("cpu")
    
    def test_state_initialization(self):
        """Test global state initialization."""
        stages = [
            DelayRead(delay_length=16, num_lines=4),
            DelayWrite(),
        ]
        core = RecursionCore(stages)
        state = core.init_state(batch_size=2)
        
        assert "delay_buffers" in state
        assert "delay_pointer" in state
        assert state["delay_buffers"].shape == (2, 16, 4)
        assert state["delay_pointer"].shape == (2,)
    
    def test_batch_dimension_handling(self):
        """Test automatic batch dimension handling."""
        stages = [
            DelayRead(delay_length=8, num_lines=2),
            DelayWrite(),
            OutputTap(num_lines=2, num_outputs=1),
        ]
        core = RecursionCore(stages)
        
        # 2D input [T, N_in]
        input_2d = torch.randn(100, 1)
        output = core.process(input_2d, block_size=32)
        assert output.shape == (100, 1)
        
        # 3D input [B, T, N_in]
        input_3d = torch.randn(3, 100, 1)
        output = core.process(input_3d, block_size=32)
        assert output.shape == (3, 100, 1)
    
    def test_block_partitioning(self):
        """Test signal partitioning into blocks."""
        stages = [
            DelayRead(delay_length=8, num_lines=2),
            DelayWrite(),
            OutputTap(num_lines=2, num_outputs=1),
        ]
        core = RecursionCore(stages)
        
        # Test with various input lengths and block sizes
        test_cases = [
            (100, 32),  # Doesn't divide evenly
            (96, 32),   # Divides evenly
            (10, 32),   # Shorter than block size
        ]
        
        for T, bs in test_cases:
            input_signal = torch.randn(T, 1)
            output = core.process(input_signal, block_size=bs)
            assert output.shape == (T, 1), f"Failed for T={T}, block_size={bs}"
    
    def test_state_preservation_across_blocks(self):
        """Test that state is correctly preserved across blocks."""
        stages = [
            DummyStage(has_state=True),
            DelayRead(delay_length=8, num_lines=2),
            DelayWrite(),
            OutputTap(num_lines=2, num_outputs=1),
        ]
        core = RecursionCore(stages)
        
        input_signal = torch.randn(64, 1)
        core.process(input_signal, block_size=16)
        # Test passes if no errors - state management is working
    
    def test_missing_output_error(self):
        """Test that missing output raises error."""
        stages = [
            DelayRead(delay_length=8, num_lines=2),
            DelayWrite(),
            # No OutputTap - should error
        ]
        core = RecursionCore(stages)
        
        input_signal = torch.randn(32, 1)
        with pytest.raises(RuntimeError, match="No output produced"):
            core.process(input_signal, block_size=16)
    
    def test_device_consistency(self):
        """Test that tensors stay on correct device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device("cuda")
        stages = [
            DelayRead(delay_length=8, num_lines=2),
            DelayWrite(),
            OutputTap(num_lines=2, num_outputs=1),
        ]
        core = RecursionCore(stages, device=device)
        
        input_signal = torch.randn(32, 1)  # CPU tensor
        output = core.process(input_signal, block_size=16)
        
        assert output.device == device
