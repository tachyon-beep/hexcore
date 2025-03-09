# Hexcore Test Plan

## 1. Introduction

This document outlines the comprehensive testing strategy for the Hexcore project - an AI assistant for Magic: The Gathering using a Mixture-of-Experts (MoE) architecture. The test plan is designed to ensure that all components of the system function correctly, both individually and as an integrated whole, and that the performance characteristics of the system meet the requirements outlined in the architecture documents.

## 2. Test Categories

The Hexcore testing framework is divided into several distinct categories, each targeting different aspects of the system:

### 2.1 Unit Tests

Unit tests focus on testing individual components in isolation to verify that each piece functions correctly on its own.

#### Data Infrastructure

- **MTGDataLoader Tests**
  - Card loading from JSON files
  - Rules loading from JSON files
  - Document creation for retrieval
  - Card and rule lookup functionality
  - Search functionality for cards and rules

#### Model Management

- **ModelLoader Tests**
  - Model loading with different quantization settings (4-bit, 8-bit)
  - Device mapping strategy tests
  - Memory allocation verification

#### Knowledge Retrieval

- **MTGRetriever Tests**
  - Document indexing
  - Basic retrieval functionality
  - Category-specific retrieval
  - Retrieval precision and recall evaluation

#### Expert Transaction Routing

- **TransactionClassifier Tests**
  - Query classification accuracy
  - Confidence score validation
  - Multi-expert activation thresholds
  - Edge case handling

#### Expert Adapters

- **ExpertAdapterManager Tests**
  - Adapter loading and initialization
  - Adapter switching functionality
  - Memory optimization during adapter swapping
  - CPU offloading verification

#### Cross-Expert Attention

- **CrossExpertAttention Tests**
  - Attention mechanism functionality
  - Information exchange between experts
  - Output combination validation
  - Edge case handling

### 2.2 Integration Tests

Integration tests verify that different components work together correctly, focusing on the interactions between modules.

#### Data and Retrieval Integration

- Test data loading and knowledge retrieval interoperation
- Verify document creation and indexing workflow
- Test end-to-end knowledge lookup processes

#### Model and Adapter Integration

- Test model loading with expert adapters
- Verify adapter switching during inference
- Test transaction classification and expert activation flow

#### Full Pipeline Integration

- End-to-end query processing tests
- Multi-expert query handling
- Knowledge integration into prompt generation
- Response generation and formatting

### 2.3 Performance Tests

Performance tests evaluate the system's efficiency, response time, and resource usage.

#### Latency Benchmarks

- Response time for different query types
- Expert switching overhead measurement
- Knowledge retrieval latency evaluation

#### Memory Usage Monitoring

- Peak memory usage during inference
- Memory consumption patterns across expert types
- GPU memory distribution between model components

#### Throughput Tests

- Maximum queries per minute under load
- Concurrent query handling capabilities
- System stability during sustained operation

### 2.4 Expert Mode Validation Tests

These tests focus on validating the specialized behavior of each expert mode to ensure they provide the intended reasoning patterns.

#### REASON Expert Tests

- Step-by-step logical reasoning validation
- Rules interpretation accuracy
- Deduction process verification

#### EXPLAIN Expert Tests

- Clarity of explanations
- Human-friendly language verification
- Technical accuracy combined with accessibility

#### TEACH Expert Tests

- Educational quality assessment
- Adaptation to different skill levels
- Conceptual breakdown evaluation

#### PREDICT Expert Tests

- Future game state simulation accuracy
- Probability estimation verification
- Strategic insight validation

#### RETROSPECT Expert Tests

- Analysis quality for past plays
- Alternative line identification
- Learning guidance validation

## 3. Test Implementation Strategy

### 3.1 Testing Frameworks and Tools

- **pytest**: Primary testing framework
- **pytest-xdist**: For parallel test execution
- **pytest-benchmark**: For performance benchmarking
- **hypothesis**: For property-based testing
- **GPUMemoryTracker**: Custom utility for GPU memory monitoring

### 3.2 Test Data Management

#### Test Card Dataset

- Create a minimal, representative subset of MTG cards for testing
- Include diverse card types, abilities, and complexities
- Ensure baseline cards like "Lightning Bolt" are included

#### Test Rule Dataset

- Create a minimal rules dataset covering core game mechanics
- Include specialized rules for testing edge cases
- Structure to match production rule format

#### Test Query Dataset

- Develop diverse queries covering all expert types
- Include simple and complex questions
- Create edge case queries for stress testing

### 3.3 Test Environment

#### Development Environment

- Local testing with targeted component tests
- Minimal data subsets for fast iteration

#### CI/CD Pipeline

- GitHub Actions for automated test execution
- Full test suite execution on pull requests
- Performance regression monitoring

#### Production Environment Testing

- Full-scale testing with complete datasets
- End-to-end validation with real-world queries
- Performance monitoring with production hardware

### 3.4 Mocking Strategy

- **Model Mocking**: Use smaller models for non-performance tests
- **GPU Mocking**: Simulate GPU functionality for CPU-only environments
- **Knowledge Base Mocking**: Use smaller, controlled knowledge bases for deterministic testing

## 4. Test Execution Plan

### 4.1 Unit Test Execution

Unit tests will be executed:

- During development (locally by developers)
- On every pull request
- Nightly on the main branch

Minimum passing threshold: 90% of all unit tests must pass.

### 4.2 Integration Test Execution

Integration tests will be executed:

- On pull requests affecting multiple components
- After merging to main branch
- Before each release

Minimum passing threshold: 100% of integration tests must pass before release.

### 4.3 Performance Test Execution

Performance tests will be executed:

- Weekly on the main branch
- Before each major release
- After significant architectural changes

Performance regression threshold: No more than 10% degradation in response time or memory usage.

## 5. Test Scenarios and Cases

### 5.1 Unit Test Cases

#### MTGDataLoader Test Cases

1. **Test Card Loading**: Verify card data loading from JSON files
2. **Test Rule Loading**: Verify rule data loading from JSON
3. **Test Card Search**: Validate card search functionality
4. **Test Rule Search**: Validate rule search functionality
5. **Test Document Creation**: Verify creation of documents for retrieval system
6. **Test Edge Cases**: Handle missing files, corrupted data, etc.

#### ModelLoader Test Cases

1. **Test 4-bit Quantization**: Verify model loading with 4-bit quantization
2. **Test 8-bit Quantization**: Verify model loading with 8-bit quantization
3. **Test Custom Device Mapping**: Validate device mapping strategy
4. **Test Memory Allocation**: Verify memory allocation across GPUs

#### Transaction Classifier Test Cases

1. **Test Query Classification**: Validate classification into expert types
2. **Test Confidence Scoring**: Verify confidence score calculation
3. **Test Multi-Expert Selection**: Validate selection of multiple experts
4. **Test Edge Case Queries**: Handle ambiguous or unusual queries

#### Expert Adapter Test Cases

1. **Test Adapter Loading**: Verify adapter loading functionality
2. **Test Adapter Switching**: Validate switching between experts
3. **Test Memory Optimization**: Verify CPU offloading of inactive adapters
4. **Test Adapter Creation**: Validate adapter creation from configuration

#### Cross-Expert Attention Test Cases

1. **Test Attention Mechanism**: Verify attention calculation
2. **Test Output Combination**: Validate combining outputs from multiple experts
3. **Test Information Exchange**: Verify information sharing between experts
4. **Test Edge Cases**: Handle single expert, missing outputs, etc.

### 5.2 Integration Test Cases

#### Pipeline Integration Test Cases

1. **Test Basic Query Flow**: Verify end-to-end processing of simple queries
2. **Test Multi-Expert Queries**: Validate processing with multiple experts
3. **Test Knowledge Integration**: Verify knowledge retrieval and integration
4. **Test Response Generation**: Validate final response generation
5. **Test Error Handling**: Verify handling of errors in any component

### 5.3 Performance Test Cases

#### Response Time Test Cases

1. **Test Simple Query Latency**: Measure response time for simple queries
2. **Test Complex Query Latency**: Measure response time for complex queries
3. **Test Sequential Query Performance**: Verify performance under sequential load
4. **Test Memory Usage Patterns**: Monitor memory usage during inference

## 6. Continuous Testing and Integration

### 6.1 CI/CD Pipeline Integration

The test suite will be integrated into a CI/CD pipeline to ensure continuous validation of code changes:

- **Pull Request Testing**: Unit tests and critical integration tests
- **Main Branch Testing**: Full test suite including performance tests
- **Release Testing**: Comprehensive testing with full datasets

### 6.2 Test Reporting

Test results will be captured and reported to provide visibility into system health:

- **Test Dashboards**: Visual dashboards showing test pass/fail metrics
- **Performance Graphs**: Charts displaying performance metrics over time
- **Coverage Reports**: Reports showing code and functionality coverage

### 6.3 Test Data Management

Test data will be managed to ensure reproducible test results:

- **Versioned Test Data**: Test datasets will be versioned alongside code
- **Synthetic Data Generation**: Tools to generate synthetic test data
- **Real-World Query Sampling**: Mechanism to capture and anonymize real queries

## 7. Test Prioritization and Roadmap

### 7.1 Priority Test Areas

The following test areas are considered highest priority:

1. **Core Inference Pipeline**: Ensures basic functionality of the system
2. **Expert Transaction Routing**: Critical for appropriate expert selection
3. **Memory Management**: Essential for operating within hardware constraints
4. **Response Quality**: Validates the ultimate quality of generated responses

### 7.2 Test Implementation Roadmap

#### Phase 1: Basic Functionality Testing

- Implement MTGDataLoader tests
- Implement ModelLoader tests
- Develop basic pipeline integration tests
- Create memory usage monitoring tests

#### Phase 2: Expert Functionality Testing

- Implement TransactionClassifier tests
- Implement ExpertAdapterManager tests
- Implement CrossExpertAttention tests
- Expand pipeline integration tests

#### Phase 3: Performance and Quality Testing

- Implement end-to-end performance tests
- Develop response quality evaluation tests
- Create stress tests
- Implement regression test suite

## 8. Test Execution Plan by Component

### 8.1 Data Layer Testing (data/\*)

| Component        | Test Type        | Priority | Automation |
| ---------------- | ---------------- | -------- | ---------- |
| MTGDataLoader    | Unit             | High     | Full       |
| RuleCompiler     | Unit             | Medium   | Full       |
| Card/Rule Search | Unit/Integration | Medium   | Full       |

### 8.2 Model Layer Testing (models/\*)

| Component             | Test Type        | Priority | Automation |
| --------------------- | ---------------- | -------- | ---------- |
| ModelLoader           | Unit/Integration | High     | Partial    |
| TransactionClassifier | Unit             | High     | Full       |
| ExpertAdapters        | Unit/Integration | High     | Partial    |
| CrossExpertAttention  | Unit             | Medium   | Full       |

### 8.3 Knowledge Layer Testing (knowledge/\*)

| Component         | Test Type   | Priority | Automation |
| ----------------- | ----------- | -------- | ---------- |
| MTGRetriever      | Unit        | High     | Full       |
| Document Indexing | Unit        | Medium   | Full       |
| Retrieval Quality | Integration | Medium   | Full       |

### 8.4 Inference Layer Testing (inference/\*)

| Component           | Test Type   | Priority | Automation |
| ------------------- | ----------- | -------- | ---------- |
| Pipeline            | Integration | High     | Full       |
| Response Generation | Integration | High     | Full       |
| Multi-Expert Mode   | Integration | Medium   | Full       |

### 8.5 End-to-End Testing

| Test Area              | Priority | Frequency | Environment |
| ---------------------- | -------- | --------- | ----------- |
| Basic Query Flow       | High     | Every PR  | CI          |
| Complex Queries        | Medium   | Daily     | CI          |
| Performance Benchmarks | Medium   | Weekly    | Dedicated   |
| Memory Usage           | High     | Daily     | CI          |

## 9. Test Implementation Guidance

### 9.1 Creating New Tests

New tests should follow these guidelines:

1. **Use pytest Fixtures**: Utilize fixtures for shared setup/teardown
2. **Include Test Documentation**: Document the purpose of each test
3. **Consider Performance**: Optimize tests for execution speed
4. **Ensure Independence**: Tests should not depend on each other
5. **Use Parameterization**: Test multiple inputs using parameterization

### 9.2 Mock and Stub Guidelines

When mocking components, follow these principles:

1. **Minimal Mocking**: Mock only what's necessary
2. **Behavior Fidelity**: Mocks should replicate real component behavior
3. **Explicit Boundaries**: Clear define mock component boundaries
4. **Validation**: Verify mock interactions match expectations

### 9.3 Test Data Guidelines

Test data should be:

1. **Representative**: Cover common and edge cases
2. **Minimal**: Use smallest dataset that validates functionality
3. **Versioned**: Track changes to test data alongside code
4. **Reproducible**: Tests should be deterministic

## 10. Implementation Priorities

Based on the current project status and existing test coverage, here is the recommended implementation priority for new tests:

### Immediate Focus (Next 2 Weeks)

1. ✅ Complete unit tests for `ExpertAdapterManager` - COMPLETED
2. ✅ Implement unit tests for `CrossExpertAttention` - COMPLETED
3. ✅ Add integration tests for multi-expert query processing - COMPLETED
4. ✅ Develop memory usage monitoring tests for inference - COMPLETED

### Short-Term Focus (Next Month)

1. Expand transaction classifier tests
2. Add performance benchmarking for different query types
3. Implement response quality evaluation framework
4. Develop stress testing for concurrent queries

### Medium-Term Focus (Next Quarter)

1. Implement comprehensive regression test suite
2. Add automated quality metrics for expert-specific responses
3. Develop advanced synthetic test data generation
4. Implement long-running stability tests

## Conclusion

This test plan provides a comprehensive framework for ensuring the quality, performance, and correctness of the Hexcore project. By systematically testing all components through a combination of unit, integration, and performance tests, we can ensure that the system meets its requirements and delivers high-quality responses to MTG-related queries.

The plan is designed to evolve alongside the project, with ongoing refinement of test cases and methodologies as the system matures. Regular execution of the test suite, both automated and manual, will provide confidence in the system's capabilities and help identify areas for improvement.
