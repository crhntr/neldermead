// Package neldermead implements the Nelder-Mead simplex optimization algorithm,
// a gradient-free method for minimizing continuous functions.
package neldermead

import (
	"cmp"
	"errors"
	"math"
	"slices"
)

const (
	DefaultAlpha         = 1.0
	DefaultBeta          = 0.5
	DefaultGamma         = 2.0
	DefaultDelta         = 0.5
	DefaultTolerance     = 1e-6
	DefaultMaxIterations = 1000
)

// Simplex is a polytope with n+1 vertices in n-dimensional space.
type Simplex struct {
	Points []Point
}

// replacePoint copies newPoint's coordinates and value into position i of the simplex.
func (s *Simplex) replacePoint(i int, newPoint Point) {
	copy(s.Points[i].X, newPoint.X)
	s.Points[i].F = newPoint.F
}

func (s *Simplex) isCollapsed(threshold float64) bool {
	if threshold == 0 {
		return false
	}
	return s.averageEdgeLength() < threshold
}

// Point holds an n-dimensional input vector X and its objective function value F.
type Point struct {
	X []float64
	F float64
}

// Objective is a function that takes an n-dimensional input vector and returns a scalar value to minimize.
type Objective = func(x []float64) float64

// Options configures the behavior of the Nelder-Mead algorithm.
// Use [NewOptions] for commonly-used defaults.
type Options struct {
	// Alpha is the reflection coefficient (commonly 1.0).
	// Controls the size of the reflection step. Must be positive.
	Alpha float64

	// Beta is the contraction coefficient (commonly 0.5).
	// Controls the size of the contraction step when the reflected point
	// does not improve the objective function value.
	// Must be in the open interval (0, 1).
	Beta float64

	// Gamma is the expansion coefficient (commonly 2.0).
	// Controls the size of the expansion step when the reflected point
	// improves the objective function value significantly.
	// Must be greater than 1.
	Gamma float64

	// Delta is the shrinkage coefficient (commonly 0.5).
	// Controls the size of the shrinking step when contraction fails
	// to improve the objective function value.
	// Must be in the open interval (0, 1).
	Delta float64

	// Tolerance is the convergence criterion.
	// The algorithm terminates when the difference in objective function values
	// between the best and worst simplex vertices is less than Tolerance.
	Tolerance float64

	// CollapseThreshold is the minimum average edge length of the simplex.
	// If the average edge length falls below this value, Run returns
	// ErrorSimplexCollapse. Set to 0 to disable collapse detection.
	CollapseThreshold float64

	// MaxIterations sets an upper bound on how long the algorithm should run to find the minima.
	MaxIterations int

	// Constraints is an optional list of per-dimension bounds.
	// If provided, the length must match the dimensionality of x0.
	// If empty, the optimization is unconstrained.
	Constraints []Constraint
}

// NewOptions returns Options with commonly-used default values.
// These should be considered a starting point; tune them for your specific problem.
func NewOptions() Options {
	return Options{
		Alpha:         DefaultAlpha,
		Beta:          DefaultBeta,
		Gamma:         DefaultGamma,
		Delta:         DefaultDelta,
		Tolerance:     DefaultTolerance,
		MaxIterations: DefaultMaxIterations,
	}
}

func (options *Options) validate() error {
	if options.Alpha <= 0 {
		return errors.New("invalid Options parameter: Alpha must be greater than 0")
	}

	if options.Beta <= 0 || options.Beta >= 1 {
		return errors.New("invalid Options parameter: Beta must be in the range (0, 1)")
	}

	if options.Gamma <= 1 {
		return errors.New("invalid Options parameter: Gamma must be greater than 1")
	}

	if options.Delta <= 0 || options.Delta >= 1 {
		return errors.New("invalid Options parameter: Delta must be in the range (0, 1)")
	}

	if options.Tolerance <= 0 {
		return errors.New("invalid Options parameter: Tolerance must be greater than 0")
	}

	if options.MaxIterations <= 0 {
		return errors.New("invalid Options parameter: MaxIterations must be greater than 0")
	}

	for _, constraint := range options.Constraints {
		err := constraint.validate()
		if err != nil {
			return err
		}
	}

	return nil
}

// Constraint defines lower and upper bounds for a single dimension.
type Constraint struct {
	Min, Max float64
}

func (c *Constraint) validate() error {
	if math.IsNaN(c.Min) || math.IsNaN(c.Max) {
		return errors.New("constraint value for Min and Max must be valid numbers")
	}

	if math.IsInf(c.Min, 0) || math.IsInf(c.Max, 0) {
		return errors.New("constraint value for Min and Max must not be infinite")
	}

	if c.Max < c.Min {
		return errors.New("constraint value for Min must be less than Max")
	}

	if c.Min == c.Max {
		return errors.New("constraint value for Min must not be equal to Max")
	}

	return nil
}

func (options *Options) validateX0(x0 []float64) error {
	if len(x0) == 0 {
		return errors.New("invalid initial x parameter: x0 must not be empty")
	}
	if len(options.Constraints) != 0 && len(options.Constraints) != len(x0) {
		return errors.New("invalid options: The number of constraints must match the length of x0")
	}
	if len(options.Constraints) != 0 {
		for i, x := range x0 {
			if x < options.Constraints[i].Min || x > options.Constraints[i].Max {
				return errors.New("invalid initial x parameter: x0 must satisfy the constraints")
			}
		}
	}
	return nil
}

// Run minimizes the objective function f starting from initial guess x0.
// It returns the best Point found (input vector and objective value).
//
// The algorithm terminates when the difference in objective function values
// between the best and worst simplex vertices falls below Tolerance, or
// when MaxIterations is reached.
//
// Run returns an error if options are invalid, x0 violates constraints,
// or the simplex collapses (see ErrorSimplexCollapse).
func Run(f Objective, x0 []float64, options Options) (Point, error) {
	if err := options.validate(); err != nil {
		return Point{}, err
	}
	if err := options.validateX0(x0); err != nil {
		return Point{}, err
	}

	simplex := createSimplex(x0, options.Constraints)

	// Evaluate the objective function at each vertex of the initial simplex.
	for i := 0; i < len(simplex.Points); i++ {
		simplex.Points[i].F = f(simplex.Points[i].X)
	}

	sortSimplex(simplex)

	// Pre-allocate a single buffer for the candidate points and centroid
	// to avoid per-iteration allocations. Each sub-slice has capacity
	// capped to prevent overlapping writes.
	n := len(x0)
	pointBuf := make([]float64, n*4)
	var (
		reflectedPoint  = Point{X: pointBuf[:n:n]}
		expandedPoint   = Point{X: pointBuf[n : n*2 : n*2]}
		contractedPoint = Point{X: pointBuf[n*2 : n*3 : n*3]}
		centroid        = pointBuf[n*3:]
	)

	done := false
	for iter := 0; iter < options.MaxIterations && !done; iter++ {
		done = runIteration(f, options, centroid, simplex, reflectedPoint, expandedPoint, contractedPoint)
		if simplex.isCollapsed(options.CollapseThreshold) {
			return Point{}, ErrorSimplexCollapse{}
		}
	}
	return simplex.Points[0], nil
}

func runIteration(f Objective, options Options, centroid []float64, simplex Simplex, reflectedPoint, expandedPoint, contractedPoint Point) bool {
	clear(centroid)
	worst := len(simplex.Points) - 1

	// Convergence check: stop if best and worst objective values are within tolerance.
	if math.Abs(simplex.Points[0].F-simplex.Points[worst].F) < options.Tolerance {
		return true
	}

	// Compute the centroid of all points except the worst.
	computeCentroid(centroid, simplex, worst)

	// Reflect the worst point through the centroid.
	reflectedPoint = transformPoint(simplex.Points[worst], reflectedPoint, f, centroid, options.Alpha)

	secondWorst := len(simplex.Points) - 2
	if reflectedPoint.F < simplex.Points[secondWorst].F {
		// Reflected point is better than second-worst: try expanding further.
		expandedPoint = transformPoint(reflectedPoint, expandedPoint, f, centroid, options.Gamma)
		if expandedPoint.F < reflectedPoint.F {
			simplex.replacePoint(worst, expandedPoint)
		} else {
			simplex.replacePoint(worst, reflectedPoint)
		}
	} else {
		// Reflected point is not better than second-worst.
		if reflectedPoint.F < simplex.Points[worst].F {
			// But it is better than the worst: accept it before contracting.
			simplex.replacePoint(worst, reflectedPoint)
		}
		// Contract: move the worst point toward the centroid.
		contractedPoint = transformPoint(simplex.Points[worst], contractedPoint, f, centroid, options.Beta)
		if contractedPoint.F < simplex.Points[worst].F {
			simplex.replacePoint(worst, contractedPoint)
		} else {
			// Contraction failed: shrink the entire simplex toward the best point.
			shrinkSimplex(simplex, options.Delta)
		}
	}

	// Re-evaluate objective at all vertices (needed after shrink; redundant otherwise).
	for i := 0; i < len(simplex.Points); i++ {
		simplex.Points[i].F = f(simplex.Points[i].X)
	}
	sortSimplex(simplex)

	if len(options.Constraints) > 0 {
		ensureXAreInConstraintBounds(simplex.Points[0].X, options.Constraints)
	}
	return false
}

// createSimplex builds the initial simplex by placing one vertex at x0
// and n additional vertices each offset by +1.0 along one axis.
func createSimplex(x0 []float64, constraints []Constraint) Simplex {
	n := len(x0)
	simplex := Simplex{Points: make([]Point, n+1)}

	for i := range simplex.Points {
		simplex.Points[i].X = make([]float64, n)
	}

	// First vertex is the initial guess.
	copy(simplex.Points[0].X, x0)

	// Remaining vertices: offset the j-th coordinate by +1.0.
	for i := 1; i <= n; i++ {
		copy(simplex.Points[i].X, x0)
		simplex.Points[i].X[i-1] += 1.0
	}

	if len(constraints) > 0 {
		for i := range simplex.Points {
			ensureXAreInConstraintBounds(simplex.Points[i].X, constraints)
		}
	}

	return simplex
}

func sortSimplex(simplex Simplex) {
	slices.SortFunc(simplex.Points, func(a, b Point) int {
		return cmp.Compare(a.F, b.F)
	})
}

// computeCentroid calculates the mean position of all simplex vertices
// except the one at excludeIndex (typically the worst).
func computeCentroid(centroid []float64, simplex Simplex, excludeIndex int) {
	for i := 0; i < len(simplex.Points); i++ {
		if i != excludeIndex {
			for j := 0; j < len(simplex.Points[i].X); j++ {
				centroid[j] += simplex.Points[i].X[j]
			}
		}
	}

	for j := 0; j < len(centroid); j++ {
		centroid[j] /= float64(len(simplex.Points) - 1)
	}
}

// shrinkSimplex moves all vertices (except the best) toward the best vertex
// by factor delta. This is the "shrink" step, used when contraction fails.
func shrinkSimplex(simplex Simplex, delta float64) {
	bestPoint := simplex.Points[0]
	for i := 1; i < len(simplex.Points); i++ {
		for j := 0; j < len(simplex.Points[i].X); j++ {
			simplex.Points[i].X[j] = bestPoint.X[j] + delta*(simplex.Points[i].X[j]-bestPoint.X[j])
		}
	}
}

// transformPoint computes: result[j] = centroid[j] + coeff * (centroid[j] - source[j])
// and evaluates f at the result. Used for reflection (Alpha), expansion (Gamma),
// and contraction (Beta) by varying the coefficient.
func transformPoint(source Point, result Point, f Objective, centroid []float64, coeff float64) Point {
	for j := 0; j < len(source.X); j++ {
		result.X[j] = centroid[j] + coeff*(centroid[j]-source.X[j])
	}
	result.F = f(result.X)
	return result
}

// ensureXAreInConstraintBounds clamps each element of x to its corresponding constraint bounds.
func ensureXAreInConstraintBounds(x []float64, constraints []Constraint) {
	for i := range x {
		x[i] = max(x[i], constraints[i].Min)
		x[i] = min(x[i], constraints[i].Max)
	}
}

// ErrorSimplexCollapse is returned by Run when the simplex degenerates
// (average edge length falls below CollapseThreshold).
type ErrorSimplexCollapse struct{}

func (ErrorSimplexCollapse) Error() string { return "simplex has collapsed" }

func (s *Simplex) averageEdgeLength() float64 {
	n := len(s.Points)
	totalLength := 0.0
	count := 0

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			totalLength += distance(s.Points[i].X, s.Points[j].X)
			count++
		}
	}

	return totalLength / float64(count)
}

func distance(a, b []float64) float64 {
	sum := 0.0
	for i := 0; i < len(a); i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}