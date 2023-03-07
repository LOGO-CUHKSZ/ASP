
def combine_Solver_configs(parser):
    config = parser.parse_args()
    if config.problem=='TSP':
        if config.method=='AM':
            from NeuralSolver.TSP.AM.options import get_options
            Solver_config = get_options(parser)
        elif config.method=='POMO':
            from NeuralSolver.TSP.POMO.options import get_options
            Solver_config = get_options(parser)
    elif config.problem=='CVRP' or config.problem=='SDVRP':
        if config.method=='AM':
            from NeuralSolver.TSP.AM.options import get_options
            Solver_config = get_options(parser)
        elif config.method=='POMO':
            from NeuralSolver.CVRP.POMO.options import get_options
            Solver_config = get_options(parser)
        elif config.method=='NeuRew':
            from NeuralSolver.CVRP.neural_rewriter.src.arguments import get_arg_parser
            Solver_config = get_arg_parser('vrp', parser)
    elif config.problem=='OP' or config.problem[:5]=='PCTSP':
        from NeuralSolver.TSP.AM.options import get_options
        Solver_config = get_options(parser)
    elif config.problem == 'JSSP':
        from NeuralSolver.JSSP.L2D.options import get_options
        Solver_config = get_options(parser)
    else:
        NotImplementedError
    return Solver_config