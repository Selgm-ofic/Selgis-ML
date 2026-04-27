# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.5] - 2026-04-27

### Added

- Thread-safety documentation for `StreamingTextDataset` and `StreamingCSVDataset`
- Comprehensive warning about `__iter__` returning self and non-thread-safe behavior
- CONTRIBUTING.md with contribution guidelines
- SECURITY.md with security policy and best practices
- .github/workflows/ci.yml for continuous integration

### Changed

- **config.py**: Added comprehensive validation in `SelgisConfig.__post_init__()` including checks for:
  - `batch_size`, `eval_batch_size` must be positive
  - `max_epochs`, `learning_rate` must be positive
  - `save_total_limit` must be non-negative
  - `patience`, `spike_threshold`, `min_history_len`, `state_update_interval` must be valid

- **core.py**: 
  - Fixed `_best_metric` initialization to be dynamic based on `higher_is_better` flag
  - Replaced `print()` statements with proper `logging` module
  - Added type hints for `get_amp_context()` return type
  - Changed logging from `[INFO]`/`[WARN]` format to standard logging levels
  - Optimized loss history access using `itertools.islice` instead of deque indexing (O(1) vs O(n))

- **lr_finder.py**:
  - Fixed `_has_device_map` detection to use `bool(getattr(model, "hf_device_map", None))`
  - Fixed memory leak in `_free_saved_state()` - now properly clears all references
  - Added type hints for `_get_amp_context()` return type
  - Added `gc.collect()` after state cleanup
  - Added `gc` import

- **trainer.py**:
  - Fixed `amp_dtype` initialization order - now computed before LRFinder creation
  - Added `strict` parameter to `load_model()` signature with default `True`
  - Fixed `step_epoch` not being called when `warmup_ratio > 0` without eval
  - Improved `_has_device_map` detection logic for device_map handling
  - Replaced `print()` with `logging`
  - Added proper warning messages for PyTorch version compatibility

- **callbacks.py**:
  - Fixed `_cleanup()` to properly handle best_model directory without extra deletions
  - Fixed `_SKIP_SUFFIXES` to not include dots and use direct comparison instead of lstrip
  - Fixed pruning mask to use `>` instead of `>=` for correct sparsity
  - Fixed logging to skip step 0 (was showing loss=0.0)
  - Replaced all `print()` with proper `logging`
  - Added TYPE_CHECKING import for type hints
  - Changed logging level to use proper INFO/WARNING/DEBUG levels

- **scheduler.py**:
  - Fixed `surge_lr()` to use consistent LR calculation through `_compute_lr_after_warmup`

- **checkpointing.py**:
  - Fixed `remove_from_model()` to properly delete `_original_forward` attribute

- **loss.py**:
  - Changed `global_max` initialization to explicit `torch.full()` for clarity

- **datasets/streaming.py**:
  - Optimized buffer operations by using `deque.popleft()` instead of `list.pop(0)` (O(1) vs O(n))
  - Added comprehensive thread-safety warnings in docstrings

- **datasets/text.py**:
  - Fixed `num_proc` to be 1 on Windows (nt) to avoid multiprocessing issues

- **datasets/factory.py**:
  - Fixed `pin_memory` to check CUDA availability before enabling
  - Optimized `worker_init_fn` to only be created when `num_workers > 0`
  - Fixed `_resolve_collate_fn` to handle property exceptions safely
  - Replaced `print()` with `logging`
  - Improved warning messages for streaming datasets

- **datasets/config.py**:
  - Added `Callable` to imports (was missing, caused NameError)

### Fixed

**Total: 44 bugs fixed (22 from BUGS_v2.md + 22 from AUDIT_SELGIS.md)**

#### Critical Issues (7):
- **[C-1]** `streaming.py:116` - IndentationError causing module import failure
- **[C-2]** `callbacks.py:92` + `trainer.py:264` - Early stopping never triggers (`should_stop` vs `_should_stop` mismatch)
- **[C-3]** `test_selgis.py:154` - Hardcoded `device='cuda'` causing test failures on CPU-only systems
- **[C-4]** `trainer.py:98` - AttributeError when using `lr_finder_enabled=True` (amp_dtype accessed before selgis creation)
- **[C-5]** `trainer.py:202,510` - TypeError when loading PEFT checkpoints (strict parameter missing)
- **[C-6]** `datasets/config.py:197` - NameError for Callable (not imported)
- **[C-7]** `core.py:60,436` - Best metric tracking broken for loss-like metrics (higher_is_better=False)

#### High Priority Issues (13):
- **[H-1]** `trainer.py:95` - LRFinder missing `save_optimizer_state` and `device` parameters
- **[H-2]** `lr_finder.py:207` - Dead code check after `loss.backward()` execution
- **[H-3]** `callbacks.py:129` - EarlyStoppingCallback overwriting HistoryCallback's training_history.json
- **[H-4]** `trainer.py:194` - PEFT checkpoint resume with `strict=True` causing RuntimeError
- **[H-5]** `text.py:470` - HFTextDataset batched tokenizer iterating over dict keys instead of values
- **[H-6]** `scheduler.py:199,215-228` - `reduce_lr()` and `surge_lr()` applying factor twice causing LR desynchronization
- **[H-7]** `core.py:431` + `trainer.py:254-260` - `step_epoch` not called when `warmup_ratio > 0`
- **[H-8]** `lr_finder.py:53` - Device map detection failing when `hf_device_map=None`
- **[H-9]** `callbacks.py:261-267` - Checkpoint cleanup deleting extra checkpoints beyond limit
- **[H-10]** `text.py:492` - Windows multiprocessing crash with `num_proc=4`
- **[H-11]** `factory.py:331` - CPU-only systems crashing with `pin_memory=True`
- **[H-12]** `trainer.py:73` - `_has_device_map` returning True when `hf_device_map=None`
- **[H-13]** `trainer.py:471` - `preds=None` being appended to `all_preds` list

#### Medium Priority Issues (13):
- **[M-1]** `callbacks.py:453,513` - _SKIP_SUFFIXES with dots causing incorrect pruning logic
- **[M-2]** `streaming.py:135,312` - O(n) performance in streaming datasets (list.pop(0))
- **[M-3]** `core.py:321-327,319` - O(n) deque indexing in hot path
- **[M-4]** `callbacks.py:366` - Misleading log output at step 0 showing loss=0.0
- **[M-5]** `callbacks.py:567` - Pruning not achieving target sparsity due to >= comparison
- **[M-6]** `checkpointing.py:276` - Gradient checkpointing leaving _original_forward attribute
- **[M-7]** `factory.py:365-370` - Unsafe collate_fn resolution with property getters
- **[M-8]** `factory.py:407` - `generator` parameter causing TypeError with IterableDataset
- **[M-9]** `checkpointing.py:357` - Standalone `remove_gradient_checkpointing()` not working with manager
- **[M-10]** `callbacks.py` - SparsityCallback missing `output_dir` API
- **[M-11]** Memory leak in LRFinder when cleaning up saved state
- **[M-12]** Type safety issue with `get_amp_context()` return type
- **[M-13]** Hardcoded device detection for HuggingFace models with device_map

#### Low Priority Issues (11):
- **[L-1]** `loss.py:120` - Unclear global_max initialization in chunked cross-entropy
- **[L-2]** `streaming.py` - Missing thread-safety documentation for streaming datasets
- **[L-3]** `callbacks.py:30+` - Trainer type hints without TYPE_CHECKING import
- **[L-4]** `test_selgis.py:~780` - `type()` hack incompatible with pickle
- **[L-5]** `factory.py:388` - worker_init_fn created unnecessarily when num_workers=0
- **[L-6]** `datasets.py` - Stub DatasetConfig not matching real implementation
- **[L-7]** `trainer.py:73` - Redundant `or` in `_has_device_map` check
- **[L-8]** `callbacks.py` - ~40 lines of duplicated code in `on_train_end` methods
- **[L-9]** `streaming.py` - Not thread-safe, `__iter__` returns self
- **[L-10]** `test_selgis.py` - `type()` hack incompatible with pickle
- **[L-11]** `pyproject.toml` - Invalid SPDX license identifier

### Performance

- Streaming datasets now use O(1) deque operations instead of O(n) list operations
- Loss history access optimized from O(n) to O(1) using itertools.islice
- Reduced unnecessary function creation when num_workers=0

### Documentation

- Added explicit thread-safety warnings for IterableDataset classes
- Clarified that `__iter__` returns self and implications for concurrent use
- Added usage examples with proper DataLoader configuration

## [0.2.4] - 2024-XX-XX

### Added

- Self-healing training framework with automatic NaN/Inf detection
- Rollback mechanism to last stable state
- Loss spike detection with configurable threshold
- CPU offload for optimizer states (40% VRAM savings)
- 4-bit and 8-bit quantization support via BitsAndBytes
- Learning rate finder (Leslie Smith method)
- Learning rate scheduler with warmup, cosine, cosine restart, linear, polynomial modes
- Final surge mechanism for plateau escape
- Gradient clipping (norm and value)
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Checkpoint management with rotation
- Streaming dataset support for large files (>100GB)
- Memory-mapped file support for fast data loading
- TextDataset, ImageDataset, MultimodalDataset, StreamingTextDataset support
- CustomDataset wrapper for any PyTorch Dataset
- Dataset factory with unified API
- CLI interface (`selgis device`, `selgis train`, `selgis test`)
- 16 comprehensive tests covering all components

### Changed

- Package name: selgis (from selgis-ml)
- Default `lr_finder_enabled` changed from True to False
- Added `learning_rate` field to SelgisConfig
- Added `primary_metric` field for early stopping
- Added `trust_remote_code` field for model loading security
- Added `device_map` field for model distribution
- Optimizer state is now cleared on rollback
- LRFinder state cleanup now uses dictionary reassignment instead of clear()

### Removed

- Deprecated checkpoint formats

### Security

- Default `trust_remote_code=False` for HuggingFace model loading
- Default `weights_only=True` for checkpoint loading
- Security warnings for untrusted checkpoint loading

## [0.2.3] - 2024-XX-XX

### Added

- Initial PyPI release
- TransformerTrainer for HuggingFace Transformers
- PEFT/LoRA support
- QLoRA (4-bit quantization + LoRA)
- Dataset configuration system
- Comprehensive API documentation

### Fixed

- Various bug fixes and improvements

## [0.2.0] - 2024-XX-XX

### Added

- Initial release candidate
- Trainer class for PyTorch models
- SelgisCore for training protection
- Callback system
- Basic dataset support

---

## Upgrade Guide

### From 0.2.x to 0.3.0

1. If you use LRFinder, ensure it's explicitly enabled:
   ```python
   config = SelgisConfig(lr_finder_enabled=True)  # Now explicit
   ```

2. If you use checkpoint loading with trust_remote_code:
   ```python
   config = TransformerConfig(trust_remote_code=True)  # Now explicit
   ```

3. Checkpoint format has changed - old checkpoints may need re-creation

### From 0.1.x to 0.2.0

1. Update imports:
   ```python
   # Old
   from selgis import Trainer
   
   # New
   from selgis import Trainer, TransformerTrainer
   ```

2. Configuration has changed - review config fields

---

## Deprecation Notices

### Future Removals

- Python 3.9 support (planned for 0.4.0)
- Legacy checkpoint format (planned for 0.4.0)

---

## Known Issues

### 0.2.4

- LLM fine-tuning tests are mocked due to VRAM requirements
- Some tests may fail on systems without CUDA

### Workarounds

- For CUDA issues: Use CPU fallback (`device="cpu"`)
- For memory issues: Enable CPU offload and quantization
- For test failures: Check system requirements

---

## Contributing

See CONTRIBUTING.md for contribution guidelines.

## License

Apache License 2.0 - see LICENSE file for details.