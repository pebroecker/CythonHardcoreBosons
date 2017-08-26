include("xml_parameters.jl")

seed_time = @elapsed rand(100, 100) * rand(100, 100)
seed = Int(round(seed_time * 1e10))
srand(seed)

L = 8
W = L
betas = 1./collect(0.1:0.1:10)
betas = [5, 10, 15, 20]
betas = [20]
len = length(betas)

for beta in betas
   for Delta in 0.1:0.1:7.0
       mkpath("jobs/L_$(L)_W_$(W)/L_$(L)_W_$(W)_beta_$(beta)/L_$(L)_W_$(W)_beta_$(beta)_Delta_$(Delta)")
       cd("jobs/L_$(L)_W_$(W)/L_$(L)_W_$(W)_beta_$(beta)/L_$(L)_W_$(W)_beta_$(beta)_Delta_$(Delta)")

       for h in -13.0:0.1:13.0    
            prefix = "L_$(L)_W_$(W)_beta_$(beta)_Delta_$(Delta)_h_$(h)"
            try
                mkdir(prefix)
    	    catch end
            cd(prefix)

            p = Dict{Any, Any}()
            p["L"] = L
            p["W"] = W
            p["Delta"] = Delta
            p["EPS"] = 0.1
            p["h"] = h
            p["BETA"] = [beta]
            p["THERMALIZATION"] = 2^13
            p["SWEEPS"] = 2^14
            p["SEED"] = convert(Int, round(rand() * 1e4))

            parameters2xml(p, prefix)
            cd("..")
        end
    end
end
