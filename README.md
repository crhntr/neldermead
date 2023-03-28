# A Nelder-Mead implementation in Go [![Go Reference](https://pkg.go.dev/badge/github.com/crhntr/neldermead.svg)](https://pkg.go.dev/github.com/crhntr/neldermead)

Nelder-Mead Optimization Algorithm in Go

This repository contains a Go implementation of the Nelder-Mead optimization algorithm, which is a gradient-free
optimization method for continuous, possibly non-convex, and noisy functions in low to moderate dimensions.

The algorithm uses a simplex, a polytope with n+1 vertices in n-dimensional space, to explore the search space. The
algorithm iteratively updates the simplex by reflecting, expanding, contracting, or shrinking it, based on the values of
the objective function at its vertices. The algorithm terminates when the difference in the objective function values
between the best and worst points in the simplex falls below the specified tolerance or when the maximum number of
iterations is reached.

## Installation

This package can be installed using Go modules with the following command:

```sh
go get github.com/crhntr/neldermead
```

## Usage

[Try it in the Go playground](https://go.dev/play/p/iLh2VgurPnf)

The Rosenbrock function is a well-known test function for optimization algorithms, and its global minimum is at (1, 1),
where the function value is 0. The Run function takes the Rosenbrock function as the objective function to minimize, the
initial guess x0 as the starting point, and the Options struct with the desired parameters for the Nelder-Mead algorithm.

The algorithm found a solution close to the global minimum with a very small objective function value.

The output of this program should be:

```
Found minimum at x = [3 10], f(x) = 17.00
```

```go
package main

import (
	"fmt"
	"math"
	
	"github.com/crhntr/neldermead"
)

func main() {
	// Define the objective function to be minimized.
	objective := func(x []float64) float64 {
		// y = x^3
		return math.Pow(x[0], 3) - x[1]
	}

	// Define the starting point for the optimizer.
	x0 := []float64{4, 5}

	// Define the optimization options.
	options := neldermead.NewOptions()
	// (optional) add constraints
	options.Constraints = []neldermead.Constraint{
		{Min: 3, Max: 5},
		{Min: 0, Max: 10},
	}

	// Run the optimizer.
	result, err := neldermead.Run(objective, x0, options)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Found minimum at x = %v, f(x) = %.2f\n", result.X, result.F)
}
```
