#For graph model, 
###Minimization Step########
min_dynModel = min_query.mod_dict[:f]
minGraph = min_query.mod_dict[:graph]
min_netModel = min_query.mod_dict[:u]
i = 0
#Compute lower bounds
sym = min_query.problem.varList[1]
v = min_query.var_dict[sym][end][1]
dv = min_query.var_dict[sym][2][1]
next_v_l = v + min_query.dt*dv


@objective(minGraph, Min, next_v_l)
set_optimizer(minGraph, Gurobi.Optimizer)
#To access the backend, i
minGraph.is_dirty = false
has_nl_objective(minGraph)
has_objective(minGraph)
Plasmo._set_graph_objective(minGraph)
objective_function(minGraph)

graph_backend = JuMP.backend(minGraph)
MOIU.state(graph_backend)
MOIU.attach_optimizer(graph_backend)
minGraph.is_dirty = false
Plasmo.has_nlp_data(minGraph)
Plasmo.has_nl_objective(minGraph)
Plasmo._set_graph_objective(minGraph)
Plasmo._set_backend_objective(minGraph)
MOI.optimize!(graph_backend)

typeof(graph_backend.optimizer.model)


#########Inside the Gurobi Backend######
graph_backend.optimizer.model.objective_sense
graph_backend.optimizer.model.last_constraint_index

graph_backend.optimizer.model.name_to_variable
graph_backend.optimizer.model.variable_info
#Constraints 
graph_backend.optimizer.model.name_to_constraint_index

graph_backend.optimizer.model.affine_constraint_info
graph_backend.optimizer.model.quadratic_constraint_info
graph_backend.optimizer.model.sos_constraint_info
graph_backend.optimizer.model.indicator_constraint_info

constrList = []
for i=1:graph_backend.optimizer.model.last_constraint_index
    push!(constrList, graph_backend.optimizer.model.affine_constraint_info[i])
end

constrList
######################################

#For flat models
###########TEST: Getting to the Backend###############
query2 = deepcopy(query)
query2.ntime = 1

query2.problem.bounds = query2.problem.bound_func(query2.problem, plotFlag=true)
query2.var_dict = Dict{Symbol,JuMP.Vector{VariableRef}}()
query2.mod_dict = Dict{Symbol,JuMP.Model}()

encode_dynamics!(query2)

#Encode the controller if it exists
if !isnothing(query2.network_file)
    encode_control!(query2)
end

t_idx = nothing
########################################
    stateVar = query2.problem.varList
    trueInp = []
    trueOut = []
    stateVarTimed = Any[]
    
    #Compute true input and output variables 
    i = 1
    for sym in stateVar
        if !isnothing(t_idx)
            #Account for symbolic case where dynamics are timed
            sym_timed = Meta.parse("$(sym)_$t_idx")
            input_vars = query2.var_dict[sym_timed][1]
            output_vars = query2.var_dict[sym_timed][2]
            push!(stateVarTimed, sym_timed)
        else   
            input_vars = query2.var_dict[sym][1]
            output_vars = query2.var_dict[sym][2]
        end

        #TODO: To this more cleverly
        #NOTE: Done, avoids the need for different cases
        push!(trueInp, input_vars...)
        push!(trueOut, output_vars...)
    end
    
    integration_map = query2.problem.update_rule(trueInp, trueOut)
    
    timestep_nplus1_vars = GenericAffExpr{Float64,VariableRef}[]
    lows = Array{Float64}(undef, 0)
    highs = Array{Float64}(undef, 0)

    #Loop over symbols with OVERT approximations to compute reach steps
    sym = stateVar[1]
    # for sym in stateVar
        #Account for symbolic case with timed dynamics
        if !isnothing(t_idx)
            symTimed = Meta.parse("$(sym)_$t_idx")
            input_vars = query2.var_dict[symTimed][1]
        else
            input_vars = query2.var_dict[sym][1]
        end
        mipModel = query2.mod_dict[sym]
        #TEST: remove
        v = input_vars[1]
        # for v in input_vars 
            # if v in trueInp
                dv = integration_map[v]
                next_v = v + query2.dt*dv
                push!(timestep_nplus1_vars, next_v)
                # @objective(mipModel, Min, next_v)
                @objective(mipModel, Max, dv)
                JuMP.optimize!(mipModel)
                termination_status(mipModel)
                @assert termination_status(mipModel) == MOI.OPTIMAL
                objective_bound(mipModel)
                push!(lows, objective_value(mipModel))
                @objective(mipModel, Max, next_v)
                JuMP.optimize!(mipModel)
                @assert termination_status(mipModel) == MOI.OPTIMAL
                objective_bound(mipModel)
                push!(highs, objective_value(mipModel))
    #         end
    #     end
    # end
    #NOTE: Hyperrectangle can plot in higher dimensions as well
    reacheable_set = Hyperrectangle(low=lows, high=highs)
