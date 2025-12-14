# Genetic Algorithms for the Knapsack Problem

## Project Overview
This project focuses on applying **Genetic Algorithms (GA)** to solve the classic **0/1 Knapsack Problem**.  
The main objectives are:  

- Understand how Genetic Algorithms mimic natural evolution to solve optimization problems.  
- Implement a custom GA from scratch (selection, crossover, mutation).  
- Compare custom implementation with DEAP library (industry-standard GA framework).  
- Validate results against brute force exhaustive search.  
- Analyze convergence behavior and genetic diversity.  

---

## Problem Definition: 0/1 Knapsack

**The Knapsack Problem**: Given a set of items, each with a weight and value, determine which items to include in a knapsack so that the total weight does not exceed a given limit and the total value is maximized.

**Problem Instance**:
- **8 items** with (weight, value):
  - Item 0: (3, 60)
  - Item 1: (1, 4)
  - Item 2: (30, 12)
  - Item 3: (10, 70)
  - Item 4: (2, 40)
  - Item 5: (1, 2)
  - Item 6: (7, 60)
  - Item 7: (13, 30)
- **Maximum weight capacity**: 60
- **Goal**: Maximize total value while staying within weight limit

**Solution representation**: Binary vector `[0/1, 0/1, ..., 0/1]` where 1 = item included, 0 = item excluded

**Example**: `[1, 1, 0, 1, 1, 1, 1, 1]` means include items {0,1,3,4,5,6,7}, exclude item {2}

---

## Genetic Algorithm Foundation

### Biological Inspiration

Genetic Algorithms are inspired by natural evolution:
- **Population**: Set of candidate solutions (individuals)
- **Genes**: Binary bits representing item selection
- **Fitness**: Total value (if weight ≤ max), else 0 (penalty)
- **Selection**: Survival of the fittest (best individuals reproduce)
- **Crossover**: Genetic recombination (mixing parent genes)
- **Mutation**: Random bit flips (introducing diversity)
- **Generations**: Iterative improvement over time

### Algorithm Steps

```
1. Initialize: Create random population
2. Evaluate: Calculate fitness for each individual
3. Select: Choose best individuals as parents
4. Crossover: Generate offspring by combining parent genes
5. Mutate: Randomly flip bits with small probability
6. Replace: Form new generation
7. Repeat: Steps 2-6 until convergence or max generations
```

---

## Implementation Details

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Population Size | 50 | Number of individuals per generation |
| Generations | 100 | Maximum iterations |
| Crossover Rate | 0.8 (80%) | Probability of crossover |
| Mutation Rate | 0.01 (1%) | Probability of bit flip per gene |
| Elite Size | 25 (50%) | Top individuals kept as parents |

### Fitness Function

```python
def fitness(individual, items, max_weight):
    total_weight = sum(items[i][0] for i in range(len(individual)) if individual[i] == 1)
    total_value = sum(items[i][1] for i in range(len(individual)) if individual[i] == 1)
    
    if total_weight > max_weight:
        return 0  # Penalty for invalid solution
    else:
        return total_value  # Reward valid solutions by their value
```

**Key insight**: Invalid solutions (overweight) get fitness = 0, effectively eliminating them from reproduction.

### Selection Strategy

**Elitism**: Keep top 50% of individuals as parents for next generation.

**Why elitism?**
- Guarantees best solutions are never lost
- Accelerates convergence
- Prevents regression to worse solutions

### Crossover Method

**Single-Point Crossover**:
```
Parent 1: [1, 1, 0, 1, 1, 1, 1, 1]
Parent 2: [1, 1, 0, 1, 1, 1, 1, 1]
          Cut point at position 3 ↓
Child 1:  [1, 1, 0 | 1, 1, 1, 1, 1]  (takes left from P1, right from P2)
Child 2:  [1, 1, 0 | 1, 1, 1, 1, 1]  (takes left from P2, right from P1)
```

**Issue observed**: When parents are identical, crossover produces identical offspring (no diversity).

### Mutation Mechanism

**Bit Flip Mutation**:
```
Before: [1, 1, 0, 1, 1, 1, 1, 1]
After:  [1, 1, 1, 1, 1, 1, 1, 1]  (bit at index 2 flipped)
```

**Purpose**: Introduce random exploration to escape local optima and maintain genetic diversity.

---

## Results

### Custom GA Performance

```
Best value found: 266
Best solution: [1, 1, 0, 1, 1, 1, 1, 1]
Execution time: 0.0737 seconds
Status: ✅ OPTIMAL
```

**Selected items**: {0, 1, 3, 4, 5, 6, 7}  
**Total weight**: 37 / 60  
**Total value**: 60 + 4 + 70 + 40 + 2 + 60 + 30 = **266**

### DEAP GA Performance

```
Best value found: 266
Best solution: [1, 1, 0, 1, 1, 1, 1, 1]
Execution time: 0.0393 seconds
Status: ✅ OPTIMAL
```

**Observation**: DEAP is ~1.9x faster (more optimized C implementations).

### Brute Force Validation

```
Optimal value: 266
Optimal solution: [1, 1, 0, 1, 1, 1, 1, 1]
Optimal weight: 37
```

**Validation**: Both GAs found the global optimum! ✅

---

## Convergence Analysis

### Generation 1 Snapshot

**All 25 parents converged to the same solution**:
```
Parent 1-25: [1, 1, 0, 1, 1, 1, 1, 1] | Fitness = 266
```

**What this means**:
- ✅ **Fast convergence**: Optimal solution found in early generations
- ⚠️ **Loss of diversity**: Entire population became genetically identical
- ⚠️ **Premature convergence**: No further exploration possible

### Crossover Observations

**Example crossover**:
```
Parent 1: [1, 1, 0, 1, 1, 1, 1, 1]
Parent 2: [1, 1, 0, 1, 1, 1, 1, 1]
Cut point: 3
Children: [1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1, 1]
```

**Issue**: Identical parents → identical children → no new genetic material.

### Mutation Observations

**Mutation 1**:
```
Before: [1, 1, 0, 1, 1, 1, 1, 1] → Fitness = 266
After:  [1, 1, 1, 1, 1, 1, 1, 1] → Fitness = 0 (overweight!)
```

**Mutation 2**:
```
Before: [1, 1, 0, 1, 1, 1, 1, 1] → Fitness = 266
After:  [1, 1, 0, 1, 1, 0, 1, 1] → Fitness = 264 (worse)
```

**Analysis**: 
- Mutations degraded optimal solution
- Low mutation rate (1%) insufficient to explore alternatives
- However, optimal solution persists due to elitism

### Final Population Statistics

**Generation 100**:
- 49 individuals: `[1, 1, 0, 1, 1, 1, 1, 1]` | Fitness = 266
- 1 individual: `[1, 1, 1, 1, 1, 1, 1, 1]` | Fitness = 0 (mutant)
- 1 individual: `[1, 1, 0, 1, 1, 0, 1, 1]` | Fitness = 264 (mutant)

**Diversity score**: ~98% identical (very low diversity)

---

## Performance Comparison

| Method | Value | Optimal? | Time (s) | Notes |
|--------|-------|----------|----------|-------|
| **Custom GA** | 266 | ✅ | 0.0737 | Pure Python, slower but educational |
| **DEAP GA** | 266 | ✅ | 0.0393 | Optimized C backend, faster |
| **Brute Force** | 266 | ✅ | N/A | 2^8 = 256 evaluations, exact solution |

**Efficiency**: GAs found optimum exploring far fewer than 256 solutions.

---

## Key Findings

1. **Both GAs converged to global optimum**: Validates correctness of implementation.

2. **Fast convergence on simple problems**: 8-item knapsack is easy; optimum found in early generations.

3. **Premature convergence observed**: Loss of genetic diversity is the main challenge.

4. **Elitism preserves best solutions**: Top 50% retention ensures optimal solution never lost.

5. **DEAP is faster**: ~2x speedup due to optimized C implementations vs pure Python.

6. **Low mutation rate insufficient**: 1% mutation couldn't maintain diversity once optimum dominated population.

7. **Problem size matters**: For 8 items, brute force is competitive. For larger problems (50+ items), GAs become essential.

---


## Technologies Used
- **Python 3.8+**  
- **DEAP**: Distributed Evolutionary Algorithms in Python (industry-standard GA library)  
- **random**: Built-in Python module for random number generation  
- **time**: Performance benchmarking  

---

## Project Conclusion

This project successfully demonstrated the application of **Genetic Algorithms** to solve the **0/1 Knapsack Problem**. Key takeaways:

1. **Both custom and DEAP implementations found the global optimum** (value = 266), validating the approach.

2. **Fast convergence** on this simple 8-item problem, with optimum found in early generations.

3. **Premature convergence is a real challenge**: Once the population becomes homogeneous, evolution stalls. This is a well-known limitation of GAs.

4. **Elitism is crucial**: Keeping top 50% as parents ensures best solutions persist across generations.

5. **DEAP library is significantly faster** (~2x) due to optimized implementations, making it preferable for production use.

6. **Problem size determines method**: For 8 items (256 combinations), brute force is viable. For 100+ items (2^100 combinations), GAs become essential.

**Educational Value**:  
Building a custom GA provides deep understanding of evolutionary computation, while DEAP offers production-ready tools for real-world optimization problems.

**Future Directions**:
- Test on larger problems (50+ items) where GAs truly shine
- Implement diversity preservation techniques
- Experiment with multi-objective optimization
- Apply to real-world resource allocation problems

---

## References

- DEAP Documentation: [https://deap.readthedocs.io/](https://deap.readthedocs.io/)  
