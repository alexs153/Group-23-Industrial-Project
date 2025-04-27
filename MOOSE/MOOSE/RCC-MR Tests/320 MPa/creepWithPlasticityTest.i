#
# This test is Example 2 from "A Consistent Formulation for the Integration
#   of Combined Plasticity and Creep" by P. Duxbury, et al., Int J Numerical
#   Methods in Engineering, Vol. 37, pp. 1277-1295, 1994.
#
# The problem is a one-dimensional bar which is loaded from yield to a value of twice
#   the initial yield stress and then unloaded to return to the original stress. The
#   bar must harden to the required yield stress during the load ramp, with no
#   further yielding during unloading. The initial yield stress (sigma_0) is prescribed
#   as 20 with a plastic strain hardening of 100. The mesh is a 1x1x1 cube with symmetry
#   boundary conditions on three planes to provide a uniaxial stress field.a
#
#  In the PowerLawCreep model, the creep strain rate is defined by:
#
#   edot = A(sigma)**n * exp(-Q/(RT)) * t**m
#
#   The creep law specified in the paper, however, defines the creep strain rate as:
#
#   edot = Ao * mo * (sigma)**n * t**(mo-1)
#      with the creep parameters given by
#         Ao = 1e-7
#         mo = 0.5
#         n  = 5
#
#   thus, input parameters for the test were specified as:
#         A = Ao * mo = 1e-7 * 0.5 = 0.5e-7
#         m = mo-1 = -0.5
#         n = 5
#         Q = 0
#
#   The variation of load P with time is:
#       P = 20 + 20t      0 < t < 1
#       P = 40 - 40(t-1)  1 < t 1.5
#
#  The analytic solution for total strain during the loading period 0 < t < 1 is:
#
#    e_tot = (sigma_0 + 20*t)/E + 0.2*t + A * t**0.5  * sigma_0**n * [ 1 + (5/3)*t +
#               + 2*t**2 + (10/7)*t**3 + (5/9)**t**4 + (1/11)*t**5 ]
#
#    and during the unloading period 1 < t < 1.5:
#
#    e_tot = (sigma_1 - 40*(t-1))/E + 0.2 + (4672/693) * A * sigma_0**n +
#               A * sigma_0**n * [ t**0.5 * ( 32 - (80/3)*t + 16*t**2 - (40/7)*t**3
#                                  + (10/9)*t**4 - (1/11)*t**5 ) - (11531/693) ]
#
#         where sigma_1 is the stress at time t = 1.
#
#  Assuming a Young's modulus (E) of 1000 and using the parameters defined above:
#
#    e_tot(1) = 2.39734
#    e_tot(1.5) = 3.16813
#
#
#   The numerically computed solution is:
#
#    e_tot(1) = 2.39718         (~0.006% error)
#    e_tot(1.5) = 3.15555       (~0.40% error)
#
[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
  type = GeneratedMesh
  dim = 3
  nx = 2
  ny = 2
  nz = 2
  ymin = 0.0
  xmax = 2
	xmin = 0.0
	ymax = 2
	zmin = 0.0
	zmax = 2
[]

[Physics/SolidMechanics/QuasiStatic]
  [all]
    strain = FINITE
    incremental = true
    add_variables = true
    generate_output = 'stress_yy elastic_strain_yy creep_strain_yy plastic_strain_yy'
  []
[]

[Functions]
  [top_pull]
    type = PiecewiseLinear
    x = '0    0.205906'
    y = '0   -320'
  []

  [dts]
    type = PiecewiseLinear
    x = '0        1'
    y = '1e-12   1e-12'
  []
[]

[BCs]
  [u_top_pull]
    type = Pressure
    variable = disp_y
    boundary = top
    function = top_pull
  []

  [u_bottom_fix]
    type = DirichletBC
    variable = disp_y
    boundary = bottom
    value = 0.0
  []
  [u_yz_fix]
    type = DirichletBC
    variable = disp_x
    boundary = left
    value = 0.0
  []
  [u_xy_fix]
    type = DirichletBC
    variable = disp_z
    boundary = back
    value = 0.0
  []
[]

[Materials]
  [elasticity_tensor]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 154.3e3
    poissons_ratio = 0.3
  []
  [creep_plas]
    type = ComputeCreepPlasticityStress
    tangent_operator = elastic
    creep_model = creep
    plasticity_model = plasticity
    max_iterations = 500
    relative_tolerance = 1e-8
    absolute_tolerance = 1e-8
  []
  [creep]
    type = PowerLawCreepStressUpdate
    coefficient = 4.4e-16
    n_exponent = 4.6
    m_exponent =-0.6079         #
    activation_energy = 0
    temperature = 823
  []
  [plasticity]
    type = IsotropicPlasticityStressUpdate
    yield_stress = 140.3
    hardening_constant = 1980
  []
[]


[Executioner]
  type = Transient
  solve_type = 'PJFNK'

  petsc_options_iname = '-pc_type -pc_hypre_type -ksp_gmres_restart -pc_hypre_boomeramg_strong_threshold -pc_hypre_boomeramg_agg_nl -pc_hypre_boomeramg_agg_num_paths -pc_hypre_boomeramg_max_levels -pc_hypre_boomeramg_coarsen_type -pc_hypre_boomeramg_interp_type -pc_hypre_boomeramg_P_max -pc_hypre_boomeramg_truncfactor -pc_hypre_boomeramg_print_statistics'
  petsc_options_value = 'hypre boomeramg 51 0.7 4 5 25 PMIS ext+i 2 0.3 0'

  line_search = 'none'
  
  nl_abs_tol = 1e-3
  nl_rel_tol = 1e-3

  nl_max_its = 50
  l_max_its = 50

  [./TimeStepper]
    type = ConstantDT
    dt = 0.5
    growth_factor = 10
  [../]

  dtmax = 5
  dtmin = 1.0e-40
  end_time = 200 
[]

[Postprocessors]
  [./stress_output]
    type = ElementAverageValue
    variable = stress_yy
  [../]
  [./creep_strain_output]
    type = ElementAverageValue
    variable = creep_strain_yy
  [../]
  [./plastic_strain_output]
    type = ElementAverageValue
    variable = plastic_strain_yy
  [../]
  [./elastic_strain_output]
    type = ElementAverageValue
    variable = elastic_strain_yy
  [../]
[]

[Outputs]
  csv=true

  [./exodus_output]
    type = Exodus
    interval = 1
  [../]
[]
