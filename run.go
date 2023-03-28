package neldermead

import (
	"errors"
	"math"

	"golang.org/x/exp/slices"
)

const (
	DefaultAlpha         = 1.0
	DefaultBeta          = 0.5
	DefaultGamma         = 2.0
	DefaultDelta         = 0.5
	DefaultTolerance     = 0.0001
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

type Options struct {
	Alpha,
	Beta,
	Gamma,
	Delta,
	Tolerance,
	CollapseThreshold float64

	MaxIterations int

	Constraints []Constraint
}

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

func (options *Options) Validate() error {
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

type Constraint struct {
	Min, Max float64
}

func (c *Constraint) validate() error {
	if math.IsNaN(c.Min) || math.IsNaN(c.Max) {
		return errors.New("fields Min and Max must be valid numbers")
	}

	if math.IsInf(c.Min, 0) || math.IsInf(c.Max, 0) {
		return errors.New("fields Min and Max must not be infinite")
	}

	if c.Min > c.Max {
		return errors.New("fields Min must be less than or equal to Max")
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

func Run(f Objective, x0 []float64, options Options) (Point, error) {
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
	slices.SortFunc(simplex.Points, func(p1, p2 Point) bool {
		return p1.F < p2.F
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

func setZero(s []float64) {
	for i := range s {
		s[i] = 0
	}
}
