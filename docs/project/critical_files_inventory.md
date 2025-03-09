# Critical Files Inventory

**Date**: March 10, 2025  
**Author**: Cline AI Assistant  
**Project**: Hexcore - MTG AI Reasoning Assistant

This document provides an inventory of all critical files in the Hexcore project, explaining their role and importance to the system. Use this as a reference to understand the project structure and ensure no orphaned code exists.

## Core Model Components

| File                                   | Purpose                                                                                 | Criticality                                 |
| -------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------- |
| `src/models/model_loader.py`           | Implements model loading with memory optimizations including 4-bit quantization support | HIGH - Entry point for initializing the LLM |
| `src/models/transaction_classifier.py` | Routes user queries to appropriate expert types based on query content                  | HIGH - Critical for expert activation       |
| `src/models/cross_expert.py`           | Implements memory-efficient attention for combining outputs from multiple experts       | HIGH - Enables expert collaboration         |
| `src/models/expert_adapters.py`        | Manages LoRA adapters for different experts with memory optimization                    | HIGH - Controls expert specialization       |
| `src/inference/pipeline.py`            | Orchestrates the complete inference workflow                                            | HIGH - Main entry point for generation      |

## Memory Management & Optimization

| File                                 | Purpose                                                         | Criticality                           |
| ------------------------------------ | --------------------------------------------------------------- | ------------------------------------- |
| `src/utils/device_mapping.py`        | Implements balanced device mapping strategy for dual GPU setups | HIGH - Critical for memory balance    |
| `src/utils/memory_management.py`     | Provides memory optimization utilities                          | HIGH - Core memory management         |
| `src/utils/gpu_memory_tracker.py`    | Tracks and analyzes GPU memory usage                            | MEDIUM - Important for debugging      |
| `src/utils/kv_cache_manager.py`      | Manages KV cache with memory constraints                        | HIGH - Improves generation efficiency |
| `src/utils/test_balanced_mapping.py` | Tests for device mapping functionality                          | MEDIUM - Validates mapping strategy   |
| `src/utils/test_model_loading.py`    | Tests for model loading performance                             | MEDIUM - Validates loading process    |

## Expert Configuration

| File                                 | Purpose                                        | Criticality                               |
| ------------------------------------ | ---------------------------------------------- | ----------------------------------------- |
| `src/utils/expert_config.py`         | Centralizes expert type configuration          | HIGH - Single source of truth for experts |
| `src/utils/gpu_utils.py`             | Helper functions for GPU operations            | MEDIUM - Utility component                |
| `src/training/classifier_trainer.py` | Implements training for transaction classifier | MEDIUM - Required for classifier training |

## Data & Knowledge Systems

| File                          | Purpose                                     | Criticality                           |
| ----------------------------- | ------------------------------------------- | ------------------------------------- |
| `src/data/mtg_data_loader.py` | Loads and processes MTG card data and rules | HIGH - Provides domain data           |
| `src/data/rule_compiler.py`   | Compiles and structures MTG rules           | HIGH - Organizes rules hierarchically |
| `src/knowledge/retriever.py`  | Implements vector-based knowledge retrieval | HIGH - Core knowledge access          |
| `data/cards.json`             | MTG card database                           | HIGH - Core domain data               |
| `data/glossary.json`          | MTG term definitions                        | MEDIUM - Supplementary knowledge      |
| `data/rules.json`             | MTG comprehensive rules                     | HIGH - Critical domain knowledge      |
| `data/test_cases.json`        | Test cases for validation                   | MEDIUM - Testing resource             |

## Tools & Utilities

| File                                     | Purpose                                 | Criticality               |
| ---------------------------------------- | --------------------------------------- | ------------------------- |
| `src/tools/__init__.py`                  | Package initialization for tools        | LOW - Structure only      |
| `src/tools/mtg_test_runner.py`           | Runs MTG-specific tests                 | MEDIUM - Testing utility  |
| `src/tools/data_utils/json_inspector.py` | Tool for analyzing JSON data structures | LOW - Development utility |
| `kv_cache_demo.py`                       | Demonstrates KV cache functionality     | LOW - Example only        |

## Tests

| File                                              | Purpose                                   | Criticality                            |
| ------------------------------------------------- | ----------------------------------------- | -------------------------------------- |
| `tests/models/test_cross_expert.py`               | Tests cross-expert attention              | HIGH - Validates core component        |
| `tests/models/test_expert_adapters.py`            | Tests expert adapter functionality        | HIGH - Validates expert management     |
| `tests/models/test_transaction_classifier.py`     | Tests transaction classification          | HIGH - Validates routing               |
| `tests/integration/test_memory_performance.py`    | Tests memory usage under load             | HIGH - Validates memory optimization   |
| `tests/integration/test_multi_expert_pipeline.py` | Tests full pipeline with multiple experts | HIGH - Validates end-to-end flow       |
| `tests/integration/test_stability.py`             | Tests system stability                    | HIGH - Validates robust operation      |
| `tests/integration/test_kv_cache_optimization.py` | Tests KV cache manager                    | MEDIUM - Validates cache functionality |
| `tests/integration/test_performance_metrics.py`   | Tests performance metrics collection      | MEDIUM - Validates monitoring          |
| `tests/utils/test_memory_monitoring.py`           | Tests memory monitoring tools             | MEDIUM - Validates monitoring          |
| `tests/utils/test_kv_cache_manager.py`            | Tests KV cache management                 | MEDIUM - Validates caching             |
| `tests/data/test_mtg_data_loader.py`              | Tests MTG data loading                    | MEDIUM - Validates data access         |
| `tests/test_harness.py`                           | Generic test harness                      | MEDIUM - Testing infrastructure        |
| `tests/test_integration.py`                       | Base integration test functionality       | MEDIUM - Testing infrastructure        |
| `tests/test_mtg_data_loader.py`                   | Alternative test for data loader          | LOW - Possibly redundant               |
| `pytest.ini`                                      | PyTest configuration                      | MEDIUM - Testing configuration         |

## Documentation

| File                                            | Purpose                                   | Criticality                |
| ----------------------------------------------- | ----------------------------------------- | -------------------------- |
| `docs/architecture/executive_summary.md`        | Executive overview of system architecture | HIGH - Project overview    |
| `docs/architecture/technical_implementation.md` | Detailed technical implementation guide   | HIGH - Technical reference |
| `docs/architecture/trainingplan.md`             | Training roadmap                          | MEDIUM - Training guidance |
| `docs/project/consolidated_status_report.md`    | Current project status                    | HIGH - Project tracking    |
| `docs/project/test_plan.md`                     | Testing strategy and plan                 | MEDIUM - Testing guidance  |
| `docs/code_review_issues.md`                    | Issues identified during code review      | MEDIUM - Development guide |

## Configuration & Setup

| File               | Purpose                            | Criticality                    |
| ------------------ | ---------------------------------- | ------------------------------ |
| `environment.yml`  | Conda environment specification    | MEDIUM - Environment setup     |
| `requirements.txt` | Python package requirements        | MEDIUM - Dependency management |
| `setup_hexcore.sh` | Project setup script               | MEDIUM - Installation          |
| `setup.py`         | Package installation configuration | MEDIUM - Installation          |
| `.gitignore`       | Git ignore patterns                | LOW - Development utility      |
| `LICENCE`          | Project license                    | MEDIUM - Legal requirement     |
| `README.md`        | Project overview                   | MEDIUM - Documentation         |

## Recently Resolved Redundancies

The following files have been reviewed and addressed:

| Original File                   | Issue                                                 | Resolution                                                                    |
| ------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------- |
| `tests/test_mtg_data_loader.py` | Duplication with `tests/data/test_mtg_data_loader.py` | Removed (was only forwarding to the new location)                             |
| `test_model_loading.py` (root)  | Duplicate of `src/utils/test_model_loading.py`        | Removed (superseded by the version in src/utils)                              |
| `data/reader.py`                | Utility script outside proper package structure       | Moved to `src/tools/data_utils/json_inspector.py` with enhanced functionality |

## Implementation Completeness

Based on the code review, the following components have these implementation states:

1. **Model Architecture**: ~90% complete

   - All core components implemented
   - Advanced dynamic features still needed

2. **Expert System**: ~95% complete

   - Classification, adaptation, and collaboration fully implemented
   - Training for adapters not yet implemented

3. **Memory Management**: ~95% complete

   - Core optimizations complete
   - Advanced monitoring features in progress

4. **Knowledge Integration**: ~85% complete

   - Basic retrieval working
   - Advanced graph integration needed

5. **Testing Infrastructure**: ~90% complete

   - Comprehensive tests implemented
   - Some advanced tests needed

6. **Pipeline Integration**: ~85% complete
   - Core functionality working
   - Production hardening needed
