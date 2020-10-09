# IMPORT BOX #
using Plots
using Plotly
using GR
using SymPy

plotly(size = (500,500))

# Defining symbolic functions f1, f2, f3.

x, y = symbols("x, y", real=true)

f1(x, y) = 2*x + 3*y + 1
f2(x, y) = x^2 + y^2 - x*y - 5
f3(x, y) = (x-5)cos(y-5) - (y-5)sin(x-5)

typeof(f1(x, y))


println(f1(x, y))
println(f2(x, y))
println(f3(x, y))

# Calculating GRADIENTS for all of the maps.
# @vars x, y
∇f1 = [diff(f1(x, y), x); diff(f1(x, y), y)]

println("=== Gradient of f2 ===")
println("df2/dx: ", diff(f2(x, y), x))
println("df2/dy: ", diff(f2(x, y), y))

println("\n=== Gradient of f3 ===")
println("df3/dx: ", diff(f3(x, y), x))
println("df3/dy: ", diff(f3(x, y), y))

println("\n\nGradient ∇f2:")
∇f2 = [diff(f2(x, y), x); diff(f2(x, y), y)]

println("\nGradient ∇f3:")
∇f3 = [diff(f3(x, y), x); diff(f3(x, y), y)]

# Calculating HESSIAN MATRICES for each:

println("=== Second Partials of f2 ===")
println("d^2(f2)/(d^2 x): ", diff(f2(x, y), x, x))
println("d^2(f2)/(dx dy): ", diff(f2(x, y), x, y))

println("\nd^2(f2)/(d^2 y): ", diff(f2(x, y), y, y))
println("d^2(f2)/(dy dx): ", diff(f2(x, y), y, x))

println("\n=== Gradient of f3 ===")
println("d^2(f3)/(d^2 x): ", diff(f3(x, y), x, x))
println("d^2(f3)/(dx dy): ", diff(f3(x, y), x, y))

println("\nd^2(f3)/(d^2 y): ", diff(f3(x, y), y, y))
println("d^2(f3)/(dy dx): ", diff(f3(x, y), y, x))

println("\n\n∇2f2:")
∇2f1 = [diff(f1(x, y), x, x) diff(f1(x, y), x, y); diff(f1(x, y), y, x) diff(f1(x, y), y, y)]

println("\n\n∇2f2:")
∇2f2 = [diff(f2(x, y), x, x) diff(f2(x, y), x, y); diff(f2(x, y), y, x) diff(f2(x, y), y, y)]

println("∇2f3:")
∇2f3 = [diff(f3(x, y), x, x) diff(f3(x, y), x, y); diff(f3(x, y), y, x) diff(f3(x, y), y, y)]

# [∇2f3[1](x=>1, y=>1) ∇2f3[2](x=>1, y=>1); ∇2f3[3](x=>1, y=>1) ∇2f3[4](x=>1, y=>1)]

# CONTOUR PLOTS FOR EACH

function get_contour_plots(sym_func, grad, POI = (1, 0), range=(-2, 3.5))
    x = range[1]:0.05:range[2]
    y = range[1]:0.05:range[2]
    p1 = Plots.contour(x, y, sym_func, fill = true, c = :acton, lw=10)
        
    x, y = symbols("x, y", real=true)
    ∇ = [grad[1](x=>POI[1], y=>POI[2]), grad[2](x=>POI[1], y=>POI[2])]
    
    Plots.plot(p1, title=string(sym_func,", POI: ",POI))
    xlabel!("x")
    ylabel!("y")
    
    Plots.quiver!([POI[1]],[POI[2]],quiver=([∇[1]],[∇[2]]), color=:white)
    
    # Tangent line drawing
    m = -1*(∇[1]/∇[2])
    b = -1*(m*POI[1])+POI[2]
    lin = m*x+b
    Plots.plot!(lin, range[1], range[2], label="Tangent", lw=5)
    Plots.ylims!(range)
    Plots.xlims!(range)
end

get_contour_plots(f1, ∇f1)

get_contour_plots(f2, ∇f2)

get_contour_plots(f3, ∇f3)

get_contour_plots(f1, ∇f1, (-0.7, 2))

get_contour_plots(f2, ∇f2, (-0.7, 2))

get_contour_plots(f3, ∇f3, (-0.7, 2))

get_contour_plots(f1, ∇f1, (2.5, -1))

get_contour_plots(f2, ∇f2, (2.5, -1))

get_contour_plots(f3, ∇f3, (2.5, -1))

# Part D: Quadratic Approximation Function
plotly(size=(500, 500, 500))
function quad_approx(sym_func, grad, hess, POI = (1, 0), range=(-2, 3.5))
    
    fx̄ = sym_func(POI[1], POI[2])
    x, y = symbols("x, y", real=true)
    ∇fx̄ = [grad[1](x=>POI[1], y=>POI[2]); grad[2](x=>POI[1], y=>POI[2])]
    ∇2fx̄ = [hess[1](x=>POI[1], y=>POI[2]) hess[2](x=>POI[1], y=>POI[2]); hess[3](x=>POI[1], y=>POI[2]) hess[4](x=>POI[1], y=>POI[2])]
    
    quad_func(x, y) = fx̄ + transpose(∇fx̄) * [x-POI[1]; y-POI[2]] + 0.5*([x-POI[1] y-POI[2]]*∇2fx̄*[x-POI[1]; y-POI[2]])[1]
    
    x = y = range[1]:0.1:range[2]
    Plots.surface(x, y, sym_func, label="Function", title="Quadratic Approx", c = :acton)
    Plots.surface!(x, y, quad_func, label="Quad Approx")
end

quad_approx(f1, ∇f1, ∇2f1)
Plots.title!("Quadratic Approximation of f1 about (1, 0)")

quad_approx(f2, ∇f2, ∇2f2)
Plots.title!("Quadratic Approximation of f2 about (1, 0)")

quad_approx(f3, ∇f3, ∇2f3)
Plots.title!("Quadratic Approximation of f3 about (1, 0)")

quad_approx(f1, ∇f1, ∇2f1, (-0.7, 2))
Plots.title!("Quadratic Approximation of f1 about (-0.7, 2)")

quad_approx(f2, ∇f2, ∇2f2, (-0.7, 2))
Plots.title!("Quadratic Approximation of f2 about (-0.7, 2)")

quad_approx(f3, ∇f3, ∇2f3, (-0.7, 2))
Plots.title!("Quadratic Approximation of f3 about (-0.7, 2)")

quad_approx(f1, ∇f1, ∇2f1, (2.5,-1))
Plots.title!("Quadratic Approximation of f1 about (2.5,-1)")

quad_approx(f2, ∇f2, ∇2f2, (2.5,-1))
Plots.title!("Quadratic Approximation of f2 about (2.5,-1)")

quad_approx(f3, ∇f3, ∇2f3, (2.5,-1))
Plots.title!("Quadratic Approximation of f3 about (2.5,-1)")


