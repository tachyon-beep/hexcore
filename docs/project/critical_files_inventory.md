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

## Adapter Training Implementation

| File                                      | Purpose                                                            | Criticality                                   |
| ----------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------- |
| `src/training/mixed_precision.py`         | Implements automatic mixed precision training with safety features | HIGH - Critical for memory-efficient training |
| `src/training/adapter_dataset.py`         | Dataset class for expert-specific data loading and preprocessing   | HIGH - Training data processing pipeline      |
| `src/training/adapter_trainer.py`         | LoRA adapter trainer with gradient accumulation and checkpointing  | HIGH - Core adapter fine-tuning functionality |
| `src/training/train_all_experts.py`       | Script to train all expert adapters with orchestration logic       | MEDIUM - Training workflow automation         |
| `src/training/expert_train_configs.py`    | Defines expert-specific training parameters and configurations     | HIGH - Training hyperparameter management     |
| `data/training/reason_examples.jsonl`     | Training data for REASON expert with step-by-step rule analysis    | HIGH - Essential training data                |
| `data/training/explain_examples.jsonl`    | Training data for EXPLAIN expert with clear concept articulation   | HIGH - Essential training data                |
| `data/training/teach_examples.jsonl`      | Training data for TEACH expert with educational structures         | HIGH - Essential training data                |
| `data/training/predict_examples.jsonl`    | Training data for PREDICT expert with probability assessments      | HIGH - Essential training data                |
| `data/training/retrospect_examples.jsonl` | Training data for RETROSPECT expert with analysis of past plays    | HIGH - Essential training data                |

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

## Knowledge Integration System

| File                                 | Purpose                                                               | Criticality                                |
| ------------------------------------ | --------------------------------------------------------------------- | ------------------------------------------ |
| `src/knowledge/query_analyzer.py`    | Analyzes queries to determine optimal retrieval strategy              | HIGH - Guides retrieval process            |
| `src/knowledge/hybrid_retriever.py`  | Implements hybrid vector + graph-based retrieval                      | HIGH - Core retrieval functionality        |
| `src/knowledge/context_assembler.py` | Selects and formats knowledge for model consumption                   | HIGH - Knowledge integration               |
| `src/knowledge/knowledge_graph.py`   | Graph-based representation of MTG entities and relationships          | HIGH - Structured knowledge representation |
| `src/knowledge/cache_manager.py`     | Caching system for knowledge retrieval with entity-based invalidation | MEDIUM - Performance optimization          |
| `src/knowledge/latency_tracker.py`   | Tracks and manages retrieval latency budgets                          | MEDIUM - Performance monitoring            |
| `examples/knowledge_system_demo.py`  | Demonstrates usage of the knowledge system                            | LOW - Example only                         |

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

## Knowledge System Tests

| File                                        | Purpose                                    | Criticality                         |
| ------------------------------------------- | ------------------------------------------ | ----------------------------------- |
| `tests/knowledge/test_query_analyzer.py`    | Tests query analysis functionality         | MEDIUM - Validates query processing |
| `tests/knowledge/test_hybrid_retriever.py`  | Tests hybrid retrieval system              | HIGH - Validates core retrieval     |
| `tests/knowledge/test_context_assembler.py` | Tests knowledge selection and formatting   | HIGH - Validates knowledge assembly |
| `tests/knowledge/test_knowledge_graph.py`   | Tests graph-based knowledge representation | HIGH - Validates knowledge model    |
| `tests/knowledge/test_cache_manager.py`     | Tests knowledge caching system             | MEDIUM - Validates caching          |
| `tests/knowledge/test_latency_tracker.py`   | Tests latency budget management            | MEDIUM - Validates performance      |
| `tests/knowledge/test_retriever.py`         | Tests base retrieval functionality         | MEDIUM - Validates retrieval        |

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
   - Training for adapters now implemented but needs final testing

3. **Memory Management**: ~95% complete

   - Core optimizations complete
   - Advanced monitoring features in progress

4. **Knowledge Integration**: ~100% complete

   - Comprehensive hybrid retrieval system implemented
   - Context assembly and latency management complete

5. **Adapter Training**: ~96% complete

   - Mixed precision training implemented
   - Dataset processing pipeline complete
   - Training configuration system complete
   - Final validation with all expert types needed

6. **Testing Infrastructure**: ~90% complete

   - Comprehensive tests implemented
   - Some advanced tests needed

7. **Pipeline Integration**: ~85% complete
   - Core functionality working
   - Production hardening needed
