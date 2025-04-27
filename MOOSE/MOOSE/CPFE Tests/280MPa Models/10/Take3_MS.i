[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
  [./gen]
	type = GeneratedMeshGenerator
	dim = 3
	nx = 10
	ny = 10
	nz = 10
# MS: Not sure what happens if you don't but I would rec specifying the mesh dimensions as well. Likewise with the type of element. Example below of lines to add:
    ymin = 0.0
    xmax = 10
	xmin = 0.0
	ymax = 10
	zmin = 0.0
	zmax = 10
#    elem_type = HEX8
  [../]
[]


[Modules/TensorMechanics/Master]
  [./all]
	add_variables = true
	strain = FINITE
	generate_output = 'stress_yy strain_yy'
  [../]
[]



[Functions]
  [./pressure]
    type = PiecewiseLinear
    x = '0   499'
# MS: units of the model are micron and MPa so should -280e-3 and -280
# MS: This is also a very fast load up. If you calculate the strain rate applied here it would be very high which will definitely affect convergence
# Look at a stress strain curve and consider what strain you would have at 280MPa. Then look what you set for gamma dot 0 for plasticity in your input values.
# Ideally the strain rate you apply in the model should be similar to gamma dot 0.
    y = '0   -280'
  [../]
  [./time_step_function]
    type = PiecewiseConstant
    x = '0 499'
	y='1 3000'
  [../]
[]

[BCs]
# Fix bottom face in all directions
  [./fix_bottom]
	type = DirichletBC
	variable = disp_y
	boundary = bottom
	value = 0.0
  [../]

# Fix X and Z displacement to prevent rigid body motion
  [./fix_x_bot]
	type = DirichletBC
	variable = disp_x
	boundary = left
	value = 0.0
  [../]

  [./fix_z_bot]
	type = DirichletBC
	variable = disp_z
	boundary = front
	value = 0.0
  [../]

# Apply 270 MPa tensile stress on the top face
  [./top_pressure]
	type = Pressure
	variable = disp_y
	boundary = top
  function=pressure # 270 MPa in Pascals, MS: model uses units of MPa
  [../]

[]


[Materials]
  [./umat]
	type = AbaqusUMATStress
	constant_properties = '0'
	plugin = '../../plugins/Bristol_CP/BRISTOL'
	num_state_vars = 72
	use_one_based_indexing = true
  [../]
[]

[UserObjects]

  # this runs the UExternalDB subroutine once at the first time step
  [uexternaldb]
    type = AbaqusUExternalDB
    plugin = '../../plugins/Bristol_CP/BRISTOL'
    execute_on = 'INITIAL'
  []
  
  # this is used to find unconverged time steps depending on
  # the UMAT output variable PNEWDT
  [time_step_size]
    type = TimestepSize
    execute_on = 'INITIAL LINEAR'
  []
  [terminator_umat]
    type = Terminator
    expression = 'time_step_size > matl_ts_min'
    fail_mode = SOFT
    execute_on = 'FINAL'
  []
[]
[Postprocessors]
  [matl_ts_min]
    type = MaterialTimeStepPostprocessor
  []
  [stress_on_load_surface]
    type = ElementAverageValue
    variable = stress_yy
    block = '0'
    use_displaced_mesh = true
  []
  [./displacement_sum]
    type = AverageNodalVariableValue
    variable = disp_y
    boundary = top
  [../]
[]


[Executioner]
  type = Transient
  solve_type = 'PJFNK'

  petsc_options_iname = '-pc_type -pc_hypre_type -ksp_gmres_restart -pc_hypre_boomeramg_strong_threshold -pc_hypre_boomeramg_agg_nl -pc_hypre_boomeramg_agg_num_paths -pc_hypre_boomeramg_max_levels -pc_hypre_boomeramg_coarsen_type -pc_hypre_boomeramg_interp_type -pc_hypre_boomeramg_P_max -pc_hypre_boomeramg_truncfactor -pc_hypre_boomeramg_print_statistics'
  petsc_options_value = 'hypre boomeramg 51 0.7 4 5 25 PMIS ext+i 2 0.3 0'

  line_search = 'none'
  
  nl_abs_tol = 1e-3
  nl_rel_tol = 1e-3

  nl_max_its = 10
  l_max_its = 50

  [TimeStepper]
  type = FunctionDT
  function=time_step_function          # Corresponding time steps
  []

  dtmax = 3000
  dtmin = 1.0e-40
  end_time = 720000
[]



[Preconditioning]
  [smp]
    type = SMP
  []
[]

[Outputs]
  csv = true
  [./out]
    type = Exodus
    interval = 100
  [../]
[]
