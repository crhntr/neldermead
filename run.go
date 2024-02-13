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

type Simplex struct {
	Points []Point
}

func (s *Simplex) replacePoint(i int, newPoint Point) {
	copy(s.Points[i].X, newPoint.X)
	s.Points[i].F = newPoint.F
	setZero(newPoint.X)
	newPoint.F = 0
}

func (s *Simplex) isCollapsed(threshold float64) bool {
	if threshold == 0 {
		return false
	}
	avgEdgeLength := s.averageEdgeLength()
	return avgEdgeLength < threshold
}

type Point struct {
	X []float64
	F float64
}

type Objective = func(x []float64) float64

// Options should be configured for your particular function and optimization problem.
// The defaults configured in NewOptions should be considered a starting point that are
// likely not well suited for your problem.
type Options struct {
	// Alpha is the reflection coefficient used in the Nelder-Mead algorithm.
	// It controls the size of the reflection step.
	// A value of 1.0 is a common choice.
	// Increasing Alpha may speed up convergence, but setting it too large may cause instability or oscillations.
	// In general, it is a positive value.
	// It should be tuned based on the specific optimization problem and the characteristics of the objective function.
	Alpha float64

	// Beta is the contraction coefficient used in the Nelder-Mead algorithm.
	// It controls the size of the contraction step when the reflected point does not improve the objective function value.
	// A common choice is 0.5, which corresponds to a step halfway between the worst point and the centroid.
	// Beta should be a positive value between 0 and 1.
	// Decreasing Beta may lead to faster convergence for functions with narrow valleys, but setting it too small may cause instability or slow convergence.
	Beta float64

	// Gamma is the expansion coefficient used in the Nelder-Mead algorithm.
	// It controls the size of the expansion step when the reflected point improves the objective function value significantly.
	// A common choice is 2.0, which doubles the step size from the centroid to the reflected point.
	// Gamma should be a positive value greater than 1.
	// Increasing Gamma can accelerate convergence for functions with wide valleys, but setting it too large may cause instability or overshooting.
	Gamma float64

	// Delta is the shrinkage coefficient used in the Nelder-Mead algorithm.
	// It controls the size of the shrinking step when the contraction step fails to improve the objective function value.
	// A common choice is 0.5, which reduces the size of the simplex by half along each edge.
	// Delta should be a positive value between 0 and 1.
	// Decreasing Delta can lead to a more thorough search in the local region, but setting it too small may cause excessive computational effort or slow convergence.
	Delta float64

	// Tolerance is the convergence criterion used in the Nelder-Mead algorithm.
	// It is the threshold for the difference in objective function values between the best and worst points in the simplex.
	// The algorithm terminates when this difference is less than or equal to Tolerance.
	// A smaller Tolerance value leads to a more accurate solution but may require more iterations to converge.
	Tolerance float64

	// CollapseThreshold is the threshold used to detect the collapse of the simplex in the Nelder-Mead algorithm.
	// It is the minimum average edge length of the simplex below which the algorithm is considered to have collapsed and returns an error.
	// A collapse may indicate that the optimization process is stuck in a degenerate region or that the chosen parameters (Alpha, Beta, Gamma, and Delta) are not suitable for the specific optimization problem.
	// If CollapseThreshold is set to 0, the collapse detection feature is disabled.
	CollapseThreshold float64

	// MaxIterations sets an upper bound on how long the algorithm should run to find the minima.
	MaxIterations int

	// Constraints is an optional list of constraints for each dimension of the optimization problem. Each Constraint
	// specifies a Min and Max value, which define the lower and upper bounds for the corresponding dimension.
	// If Constraints is not provided or is empty, the optimization problem is considered unconstrained.
	// Providing Constraints can help guide the optimization process and prevent the algorithm from exploring
	// infeasible regions of the search space. The appropriate constraints should be chosen based on the problem's
	// specific requirements and the characteristics of the objective function.
	Constraints []Constraint
}

// NewOptions should be considered a starting point that may not be suited for your optimization problem.
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
		return errors.New("invalid Options parameter: Beta must be in the range [0, 1]")
	}

	if options.Gamma <= 1 {
		return errors.New("invalid Options parameter: Gamma must be greater than 1")
	}

	if options.Delta <= 0 || options.Delta >= 1 {
		return errors.New("invalid Options parameter: Delta must be in the range [0, 1]")
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
		return errors.New("constraint value for Min not be equal to Max")
	}

	return nil
}

func (options *Options) validateX0(x0 []float64) error {
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

// Run is the main function that performs optimization using the Nelder-Mead algorithm. It takes an objective
// function 'f', an initial guess 'x0', and an Options struct that configures the behavior of the algorithm.
// Run returns the best solution found as a Point, containing the optimized input vector and the corresponding
// value of the objective function. If an error occurs during the optimization process, such as a simplex
// collapse or validation error, Run returns an error.
//
// The Nelder-Mead algorithm is a gradient-free optimization method that uses a simplex (a polytope with n+1
// vertices in n-dimensional space) to explore the search space. The algorithm iteratively updates the simplex
// by reflecting, expanding, contracting, or shrinking it, based on the values of the objective function at
// its vertices. The algorithm terminates when the difference in the objective function values between the
// best and worst points in the simplex falls below the specified Tolerance, or when the maximum number of
// iterations is reached.
//
// The Run function is suitable for optimizing continuous, possibly non-convex, and noisy functions in
// low to moderate dimensions. However, its performance may degrade as the dimensionality of the problem
// increases or if the objective function has numerous local minima or sharp features.
func Run(f Objective, x0 []float64, options Options) (Point, error) {
	if err := options.validate(); err != nil {
		return Point{}, err
	}
	if err := options.validateX0(x0); err != nil {
		return Point{}, err
	}

	simplex := createSimplex(x0, len(x0), options.Constraints)

	for i := 0; i < len(simplex.Points); i++ {
		simplex.Points[i].F = f(simplex.Points[i].X)
	}

	sortSimplex(simplex)

	var (
		n               = len(x0)
		pointBuf        = make([]float64, n*4, n*4)
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
	setZero(centroid)
	lastPointIndex := len(simplex.Points) - 1
	if math.Abs(simplex.Points[0].F-simplex.Points[lastPointIndex].F) < options.Tolerance {
		return true
	}
	computeCentroid(centroid, simplex, lastPointIndex)
	reflectedPoint = simplex.Points[lastPointIndex].reflect(reflectedPoint, f, centroid, options.Alpha)
	if reflectedPoint.F < simplex.Points[len(simplex.Points)-2].F {
		expandedPoint = reflectedPoint.reflect(expandedPoint, f, centroid, options.Gamma)
		if expandedPoint.F < reflectedPoint.F {
			simplex.replacePoint(lastPointIndex, expandedPoint)
		} else {
			simplex.replacePoint(lastPointIndex, reflectedPoint)
		}
	} else {
		if reflectedPoint.F < simplex.Points[lastPointIndex].F {
			simplex.replacePoint(lastPointIndex, reflectedPoint)
		}
		contractedPoint = simplex.Points[lastPointIndex].reflect(contractedPoint, f, centroid, options.Beta)
		if contractedPoint.F < simplex.Points[lastPointIndex].F {
			simplex.replacePoint(lastPointIndex, contractedPoint)
		} else {
			shrinkSimplex(simplex, options.Delta)
		}
	}
	for i := 0; i < len(simplex.Points); i++ {
		simplex.Points[i].F = f(simplex.Points[i].X)
	}
	sortSimplex(simplex)
	if len(options.Constraints) > 0 {
		ensureXAreInConstraintBounds(simplex.Points[0].X, options.Constraints)
	}
	return false
}

func createSimplex(x []float64, n int, constraints []Constraint) Simplex {
	simplex := Simplex{Points: make([]Point, n+1)}

	for i := range simplex.Points {
		simplex.Points[i].X = make([]float64, len(x))
	}

	for j := 0; j < n; j++ {
		simplex.Points[0].X[j] = x[j]
	}

	for i := 1; i <= n; i++ {
		for j := 0; j < n; j++ {
			if i-1 == j {
				simplex.Points[i].X[j] = x[j] + 1.0
			} else {
				simplex.Points[i].X[j] = x[j]
			}
		}
	}

	if len(constraints) > 0 {
		for i := 0; i < len(simplex.Points); i++ {
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

func shrinkSimplex(simplex Simplex, delta float64) {
	bestPoint := simplex.Points[0]
	for i := 1; i < len(simplex.Points); i++ {
		for j := 0; j < len(simplex.Points[i].X); j++ {
			simplex.Points[i].X[j] = bestPoint.X[j] + delta*(simplex.Points[i].X[j]-bestPoint.X[j])
		}
	}
}

func (p *Point) reflect(reflectedPoint Point, f Objective, centroid []float64, alpha float64) Point {
	for j := 0; j < len(p.X); j++ {
		reflectedPoint.X[j] = centroid[j] + alpha*(centroid[j]-p.X[j])
	}
	reflectedPoint.F = f(reflectedPoint.X)
	return reflectedPoint
}

func ensureXAreInConstraintBounds(x []float64, constraints []Constraint) {
	for i := range x {
		if x[i] < constraints[i].Min {
			x[i] = constraints[i].Min
		}
		if x[i] > constraints[i].Max {
			x[i] = constraints[i].Max
		}
	}
}

type ErrorSimplexCollapse struct{}

func (ErrorSimplexCollapse) Error() string { return "simplex has collapsed" }

func (s *Simplex) averageEdgeLength() float64 {
	n := len(s.Points)
	totalLength := 0.0
	count := 0

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			dist := distance(s.Points[i].X, s.Points[j].X)
			totalLength += dist
			count++
		}
	}

	return totalLength / float64(count)
}

func distance(x1, x2 []float64) float64 {
	sum := 0.0
	for i := 0; i < len(x1); i++ {
		d := x1[i] - x2[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

func setZero[T any](s []T) {
	var zero T
	for i := range s {
		s[i] = zero
	}
}
