1

   Abaqus 2024                                  Date 20-Mar-2025   Time 15:20:42
   For use by UNIVERSITY OF BRISTOL under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     S T A T I C   A N A L Y S I S


                                                                                          

     AUTOMATIC TIME CONTROL WITH -
          A SUGGESTED INITIAL TIME INCREMENT OF                0.200    
          AND A TOTAL TIME PERIOD OF                           0.200    
          THE MINIMUM TIME INCREMENT ALLOWED IS                2.000E-06
          THE MAXIMUM TIME INCREMENT ALLOWED IS                0.200    

     LINEAR EQUATION SOLVER TYPE         DIRECT SPARSE

 CONVERGENCE TOLERANCE PARAMETERS FOR FORCE    
     CRITERION FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR DISP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE FORCE                               1.000E-02
     AVERAGE FORCE     IS TIME AVERAGE FORCE    
     ALTERNATE CRIT. FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO FORCE     RELATIVE TO TIME AVRG. FORCE         1.000E-05
     CRITERION FOR RESIDUAL FORCE     WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR DISP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL FORCE     FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO FORCE     REL. TO TIME AVRG. MAX. FORCE        1.000E-05
     CRITERION FOR ZERO DISP.    RELATIVE TO CHARACTERISTIC LENGTH     1.000E-08

 CONVERGENCE TOLERANCE PARAMETERS FOR MOMENT   
     CRITERION FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR ROTATION CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE MOMENT                              1.000E-02
     AVERAGE MOMENT    IS TIME AVERAGE MOMENT   
     ALTERNATE CRIT. FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO MOMENT    RELATIVE TO TIME AVRG. MOMENT        1.000E-05
     CRITERION FOR RESIDUAL MOMENT    WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR ROTATION CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL MOMENT    FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO MOMENT    REL. TO TIME AVRG. MAX. MOMENT       1.000E-05

     VOLUMETRIC STRAIN COMPATIBILITY TOLERANCE FOR HYBRID SOLIDS       1.000E-05
     AXIAL STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS             1.000E-05
     TRANS. SHEAR STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS      1.000E-05
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P>P0          5.000E-03
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P=0.0         0.100    
     CONTACT FORCE ERROR TOLERANCE FOR CONVERT SDI=YES                 1.00    
     DISPLACEMENT COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS           1.000E-05
     ROTATION COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS               1.000E-05

 EQUILIBRIUM WILL BE CHECKED FOR SEVERE DISCONTINUITY ITERATIONS

 TIME INCREMENTATION CONTROL PARAMETERS:
     FIRST EQUILIBRIUM ITERATION FOR CONSECUTIVE DIVERGENCE CHECK              4
     EQUILIBRIUM ITERATION AT WHICH LOG. CONVERGENCE RATE CHECK BEGINS         8
     EQUILIBRIUM ITERATION AFTER WHICH ALTERNATE RESIDUAL IS USED              9
     MAXIMUM EQUILIBRIUM ITERATIONS ALLOWED                                   16
     EQUILIBRIUM ITERATION COUNT FOR CUT-BACK IN NEXT INCREMENT               10
     MAXIMUM EQUILIB. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      4
     MAXIMUM ITERATIONS FOR SEVERE DISCONTINUITIES                            50
     MAXIMUM ATTEMPTS ALLOWED IN AN INCREMENT                                  5
     MAXIMUM DISCON. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      50
     MAXIMUM CONTACT AUGMENTATIONS FOR *SURFACE BEHAVIOR,AUGMENTED LAGRANGE   50
     CUT-BACK FACTOR AFTER DIVERGENCE                                 0.2500    
     CUT-BACK FACTOR FOR TOO SLOW CONVERGENCE                         0.5000    
     CUT-BACK FACTOR AFTER TOO MANY EQUILIBRIUM ITERATIONS            0.7500    
     CUT-BACK FACTOR AFTER TOO MANY SEVERE DISCONTINUITY ITERATIONS   0.2500    
     CUT-BACK FACTOR AFTER PROBLEMS IN ELEMENT ASSEMBLY               0.2500    
     INCREASE FACTOR AFTER TWO INCREMENTS THAT CONVERGE QUICKLY        1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED                       1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DYNAMICS)            1.250    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DIFFUSION)           2.000    
     MINIMUM TIME INCREMENT RATIO FOR EXTRAPOLATION TO OCCUR          0.1000    
     MAX. RATIO OF TIME INCREMENT TO STABILITY LIMIT                   1.000    
     FRACTION OF STABILITY LIMIT FOR NEW TIME INCREMENT               0.9500    
     TIME INCREMENT INCREASE FACTOR BEFORE A TIME POINT                1.000    
     GLOBAL STABILIZATION CONTROL IS NOT USED

          PRINT OF INCREMENT NUMBER, TIME, ETC., EVERY    1  INCREMENTS

     THE MAXIMUM NUMBER OF INCREMENTS IN THIS STEP IS                     100

          LARGE DISPLACEMENT THEORY WILL BE USED

     LINEAR EXTRAPOLATION WILL BE USED

     CHARACTERISTIC ELEMENT LENGTH     0.465    

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.200    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  10ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       195.       TIME AVG. FORCE        195.    
 LARGEST SCALED RESIDUAL FORCE      -827.       AT NODE         96   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -827.    
 LARGEST INCREMENT OF DISP.         9.471E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        9.471E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       90.7    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.876E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.876E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       183.       TIME AVG. FORCE        183.    
 LARGEST SCALED RESIDUAL FORCE      1.427E+03   AT NODE       1505   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.427E+03
 LARGEST INCREMENT OF DISP.          1.32       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.         1.23       AT NODE        377   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       84.8    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.883E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.502E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 716 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       269.       TIME AVG. FORCE        269.    
 LARGEST SCALED RESIDUAL FORCE      2.837E+03   AT NODE        196   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.837E+03
 LARGEST INCREMENT OF DISP.          3.45       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.         2.13       AT NODE        417   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       125.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.879E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.924E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 720 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       218.       TIME AVG. FORCE        218.    
 LARGEST SCALED RESIDUAL FORCE       144.       AT NODE        206   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       144.    
 LARGEST INCREMENT OF DISP.          3.95       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        0.499       AT NODE         72   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       101.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.571E-36   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 721 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       217.       TIME AVG. FORCE        217.    
 LARGEST SCALED RESIDUAL FORCE       17.2       AT NODE        215   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       17.2    
 LARGEST INCREMENT OF DISP.          4.07       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        0.124       AT NODE        413   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       101.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.514E-36   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 721 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                       217.       TIME AVG. FORCE        217.    
 LARGEST SCALED RESIDUAL FORCE      0.271       AT NODE        259   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.271    
 LARGEST INCREMENT OF DISP.          4.08       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.760E-03   AT NODE         96   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       101.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.068E-37   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.200    ,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED       0.200    ,  TOTAL TIME COMPLETED        0.200    
1

   Abaqus 2024                                  Date 20-Mar-2025   Time 15:20:43
   For use by UNIVERSITY OF BRISTOL under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    2     INCREMENT     1     STEP TIME    0.00    


                        S T E P       2     V I S C O   A N A L Y S I S


                                                                                          

     AUTOMATIC TIME CONTROL WITH -
          A SUGGESTED INITIAL TIME INCREMENT OF                1.000E-06
          AND A TOTAL TIME PERIOD OF                            200.    
          THE MINIMUM TIME INCREMENT ALLOWED IS                1.000E-09
          THE MAXIMUM TIME INCREMENT ALLOWED IS                 200.    

     THE SIZE OF THE TIME INCREMENT IS CONTROLLED BY -
          THE PARAMETER  CETOL                                 1.000E-05

                              IMPLICIT TIME INTEGRATION WILL BE USED

 CONVERGENCE TOLERANCE PARAMETERS FOR FORCE    
     CRITERION FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR DISP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INIT. VALUE OF TIME AVG. FORCE     IS TIME AVG. FORCE     IN PREVIOUS STEP
     AVERAGE FORCE     IS TIME AVERAGE FORCE    
     ALTERNATE CRIT. FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO FORCE     RELATIVE TO TIME AVRG. FORCE         1.000E-05
     CRITERION FOR RESIDUAL FORCE     WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR DISP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL FORCE     FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO FORCE     REL. TO TIME AVRG. MAX. FORCE        1.000E-05
     CRITERION FOR ZERO DISP.    RELATIVE TO CHARACTERISTIC LENGTH     1.000E-08

 CONVERGENCE TOLERANCE PARAMETERS FOR MOMENT   
     CRITERION FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR ROTATION CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INIT. VALUE OF TIME AVG. MOMENT    IS TIME AVG. MOMENT    IN PREVIOUS STEP
     AVERAGE MOMENT    IS TIME AVERAGE MOMENT   
     ALTERNATE CRIT. FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO MOMENT    RELATIVE TO TIME AVRG. MOMENT        1.000E-05
     CRITERION FOR RESIDUAL MOMENT    WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR ROTATION CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL MOMENT    FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO MOMENT    REL. TO TIME AVRG. MAX. MOMENT       1.000E-05

     VOLUMETRIC STRAIN COMPATIBILITY TOLERANCE FOR HYBRID SOLIDS       1.000E-05
     AXIAL STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS             1.000E-05
     TRANS. SHEAR STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS      1.000E-05
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P>P0          5.000E-03
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P=0.0         0.100    
     CONTACT FORCE ERROR TOLERANCE FOR CONVERT SDI=YES                 1.00    
     DISPLACEMENT COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS           1.000E-05
     ROTATION COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS               1.000E-05

 EQUILIBRIUM WILL BE CHECKED FOR SEVERE DISCONTINUITY ITERATIONS

 TIME INCREMENTATION CONTROL PARAMETERS:
     FIRST EQUILIBRIUM ITERATION FOR CONSECUTIVE DIVERGENCE CHECK              4
     EQUILIBRIUM ITERATION AT WHICH LOG. CONVERGENCE RATE CHECK BEGINS         8
     EQUILIBRIUM ITERATION AFTER WHICH ALTERNATE RESIDUAL IS USED              9
     MAXIMUM EQUILIBRIUM ITERATIONS ALLOWED                                   16
     EQUILIBRIUM ITERATION COUNT FOR CUT-BACK IN NEXT INCREMENT               10
     MAXIMUM EQUILIB. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      4
     MAXIMUM ITERATIONS FOR SEVERE DISCONTINUITIES                            50
     MAXIMUM ATTEMPTS ALLOWED IN AN INCREMENT                                  5
     MAXIMUM DISCON. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      50
     MAXIMUM CONTACT AUGMENTATIONS FOR *SURFACE BEHAVIOR,AUGMENTED LAGRANGE   50
     CONSECUTIVE INCS MEETING TIME INTEG. TOL. FOR TIME INC. INCREASE          3
     CUT-BACK FACTOR AFTER DIVERGENCE                                 0.2500    
     CUT-BACK FACTOR FOR TOO SLOW CONVERGENCE                         0.5000    
     CUT-BACK FACTOR AFTER TOO MANY EQUILIBRIUM ITERATIONS            0.7500    
     CUT-BACK FACTOR AFTER TOO MANY SEVERE DISCONTINUITY ITERATIONS   0.2500    
     CUT-BACK FACTOR AFTER PROBLEMS IN ELEMENT ASSEMBLY               0.2500    
     INCREASE FACTOR AFTER TWO INCREMENTS THAT CONVERGE QUICKLY        1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED                       1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DYNAMICS)            1.250    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DIFFUSION)           2.000    
     MINIMUM TIME INCREMENT RATIO FOR EXTRAPOLATION TO OCCUR          0.1000    
     CUT-BACK FACTOR WHEN TIME INTEGRATION ACCURACY TOL. EXCEEDED     0.8500    
     MAX. RATIO OF TIME INTEG. MEASURE TO TOL. FOR TIME INC. INCREASE 0.7500    
     INCREASE FACTOR FOR TIME INCREMENT WHEN POSSIBLE                 0.8000    
     MIN. TIME INC. INCREASE FACTOR IN LINEAR TRANSIENT PROBLEMS      0.9500    
     MAX. RATIO OF TIME INCREMENT TO STABILITY LIMIT                   1.000    
     FRACTION OF STABILITY LIMIT FOR NEW TIME INCREMENT               0.9500    
     TIME INCREMENT INCREASE FACTOR BEFORE A TIME POINT                1.000    
     GLOBAL STABILIZATION CONTROL IS NOT USED

          PRINT OF INCREMENT NUMBER, TIME, ETC., EVERY    1  INCREMENTS

     THE MAXIMUM NUMBER OF INCREMENTS IN THIS STEP IS                 9999999

          LARGE DISPLACEMENT THEORY WILL BE USED

     LINEAR EXTRAPOLATION WILL BE USED

     CHARACTERISTIC ELEMENT LENGTH     0.465    

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      2.803E-02   AT NODE       1519   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.803E-02
 LARGEST INCREMENT OF DISP.         1.823E-07   AT NODE        259   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.823E-07   AT NODE        259   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.392E-02   AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.392E-02
 LARGEST INCREMENT OF DISP.         3.242E-07   AT NODE        341   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.814E-07   AT NODE        341   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.181E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -2.996E-03   AT NODE       1510   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.996E-03
 LARGEST INCREMENT OF DISP.         3.656E-07   AT NODE        341   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        7.818E-08   AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.526E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      1.108E-03   AT NODE         16   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.108E-03
 LARGEST INCREMENT OF DISP.         3.827E-07   AT NODE        341   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.669E-08   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.239E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      5.974E-04   AT NODE        373   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.974E-04
 LARGEST INCREMENT OF DISP.         3.849E-07   AT NODE        341   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        7.784E-09   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.135E-43   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      4.340E-07   AT NODE        364   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      4.340E-07
 LARGEST INCREMENT OF DISP.         3.849E-07   AT NODE        341   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.092E-09   AT NODE        373   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.890E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.439E-14

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-06,  FRACTION OF STEP COMPLETED  5.000E-09
 STEP TIME COMPLETED       1.000E-06,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -2.631E-04   AT NODE        615   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.631E-04
 LARGEST INCREMENT OF DISP.         6.291E-09   AT NODE        606   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -3.849E-07   AT NODE        341   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.928E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -2.979E-07   AT NODE       1432   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.979E-07
 LARGEST INCREMENT OF DISP.         1.295E-08   AT NODE        606   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        6.861E-09   AT NODE        598   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.484E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.113E-15

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-06,  FRACTION OF STEP COMPLETED  1.000E-08
 STEP TIME COMPLETED       2.000E-06,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.277E-06   AT NODE       1424   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.277E-06
 LARGEST INCREMENT OF DISP.         1.212E-08   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -9.431E-10   AT NODE        606   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -7.595E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   7.309E-18
          TIME INCREMENT MAY NOW INCREASE TO    2.000E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-06,  FRACTION OF STEP COMPLETED  1.500E-08
 STEP TIME COMPLETED       3.000E-06,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -5.439E-09   AT NODE       1432   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -5.439E-09
 LARGEST INCREMENT OF DISP.         2.425E-08   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.213E-11   AT NODE       1432   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.876E-17
          TIME INCREMENT MAY NOW INCREASE TO    4.000E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.000E-06,  FRACTION OF STEP COMPLETED  2.500E-08
 STEP TIME COMPLETED       5.000E-06,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -3.212E-09   AT NODE       1424   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.212E-09
 LARGEST INCREMENT OF DISP.         4.849E-08   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.353E-12   AT NODE       1432   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.153E-16
          TIME INCREMENT MAY NOW INCREASE TO    8.000E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.000E-06,  FRACTION OF STEP COMPLETED  4.500E-08
 STEP TIME COMPLETED       9.000E-06,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -4.934E-11   AT NODE        416   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.934E-11
 LARGEST INCREMENT OF DISP.         9.698E-08   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.748E-12   AT NODE         61   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.613E-16
          TIME INCREMENT MAY NOW INCREASE TO    1.600E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.000E-06,  FRACTION OF STEP COMPLETED  8.500E-08
 STEP TIME COMPLETED       1.700E-05,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.600E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -3.536E-11   AT NODE        398   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.536E-11
 LARGEST INCREMENT OF DISP.         1.939E-07   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -7.070E-12   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.845E-15
          TIME INCREMENT MAY NOW INCREASE TO    3.200E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.600E-05,  FRACTION OF STEP COMPLETED  1.650E-07
 STEP TIME COMPLETED       3.300E-05,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.200E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.138       AT NODE       1412   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.138    
 LARGEST INCREMENT OF DISP.         4.858E-06   AT NODE        266   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        4.494E-06   AT NODE        266   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      9.573E-02   AT NODE        898   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      9.573E-02
 LARGEST INCREMENT OF DISP.         1.320E-05   AT NODE        266   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        9.334E-06   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.224E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      5.080E-02   AT NODE        191   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.080E-02
 LARGEST INCREMENT OF DISP.         1.852E-05   AT NODE        266   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        6.312E-06   AT NODE        111   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.289E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      2.080E-02   AT NODE       1453   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.080E-02
 LARGEST INCREMENT OF DISP.         2.110E-05   AT NODE          5   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.139E-06   AT NODE        109   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.514E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -7.511E-03   AT NODE       1471   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -7.511E-03
 LARGEST INCREMENT OF DISP.         2.198E-05   AT NODE        110   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.006E-06   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.048E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.133E-03   AT NODE       1483   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.133E-03
 LARGEST INCREMENT OF DISP.         2.212E-05   AT NODE        110   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.441E-07   AT NODE        107   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.459E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.732E-08

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.200E-05,  FRACTION OF STEP COMPLETED  3.250E-07
 STEP TIME COMPLETED       6.500E-05,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.200E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.170       AT NODE        595   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.170    
 LARGEST INCREMENT OF DISP.         1.974E-04   AT NODE        107   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.754E-04   AT NODE        107   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.521E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -5.923E-02   AT NODE       1458   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -5.923E-02
 LARGEST INCREMENT OF DISP.         2.005E-04   AT NODE        107   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.146E-06   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    1.098E-06
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.277E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.166E-08

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.200E-05,  FRACTION OF STEP COMPLETED  4.850E-07
 STEP TIME COMPLETED       9.700E-05,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.200E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       286.       TIME AVG. FORCE        333.    
 LARGEST SCALED RESIDUAL FORCE      -1.95       AT NODE        197   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.95    
 LARGEST INCREMENT OF DISP.        -1.074E-05   AT NODE        326   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.090E-04   AT NODE        109   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.089E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.256       AT NODE       1465   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.256    
 LARGEST INCREMENT OF DISP.         6.558E-05   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        7.330E-05   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.561E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.129       AT NODE        352   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.129    
 LARGEST INCREMENT OF DISP.         6.949E-05   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.933E-06   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.011E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      1.995E-02   AT NODE        895   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.995E-02
 LARGEST INCREMENT OF DISP.         7.134E-05   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.987E-06   AT NODE        609   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION   -3.077E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.233E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.799E-09
          TIME INCREMENT MAY NOW INCREASE TO    4.800E-05

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.200E-05,  FRACTION OF STEP COMPLETED  6.450E-07
 STEP TIME COMPLETED       1.290E-04,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.800E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       292.       TIME AVG. FORCE        334.    
 LARGEST SCALED RESIDUAL FORCE      -2.00       AT NODE        194   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -2.00    
 LARGEST INCREMENT OF DISP.        -1.224E-05   AT NODE        326   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.111E-04   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.145       AT NODE        607   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.145    
 LARGEST INCREMENT OF DISP.         6.471E-05   AT NODE        106   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        6.867E-05   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.232E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -6.898E-02   AT NODE        348   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -6.898E-02
 LARGEST INCREMENT OF DISP.         6.700E-05   AT NODE        106   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.372E-06   AT NODE        432   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.779E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -3.407E-02   AT NODE       1507   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.407E-02
 LARGEST INCREMENT OF DISP.         6.906E-05   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.134E-06   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.774E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -8.855E-03   AT NODE         15   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.855E-03
 LARGEST INCREMENT OF DISP.         6.963E-05   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        6.461E-07   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.114E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.413E-09

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.800E-05,  FRACTION OF STEP COMPLETED  8.850E-07
 STEP TIME COMPLETED       1.770E-04,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.800E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       298.       TIME AVG. FORCE        335.    
 LARGEST SCALED RESIDUAL FORCE      -1.53       AT NODE        198   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.53    
 LARGEST INCREMENT OF DISP.         1.349E-05   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -5.726E-05   AT NODE        617   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.490E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -7.839E-02   AT NODE        352   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -7.839E-02
 LARGEST INCREMENT OF DISP.         5.242E-05   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.895E-05   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.660E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -3.776E-02   AT NODE       1451   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.776E-02
 LARGEST INCREMENT OF DISP.         5.338E-05   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        9.851E-07   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    4.745E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.339E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.236E-09

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.800E-05,  FRACTION OF STEP COMPLETED  1.125E-06
 STEP TIME COMPLETED       2.250E-04,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.800E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       306.       TIME AVG. FORCE        335.    
 LARGEST SCALED RESIDUAL FORCE     -0.784       AT NODE         98   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.784    
 LARGEST INCREMENT OF DISP.         2.289E-05   AT NODE        108   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -3.118E-05   AT NODE         85   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.003E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -6.268E-02   AT NODE        613   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -6.268E-02
 LARGEST INCREMENT OF DISP.         4.149E-05   AT NODE        106   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.922E-05   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.733E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -3.788E-02   AT NODE         96   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.788E-02
 LARGEST INCREMENT OF DISP.         4.272E-05   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.290E-06   AT NODE        385   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.357E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.385E-02   AT NODE        600   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.385E-02
 LARGEST INCREMENT OF DISP.         4.402E-05   AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.320E-06   AT NODE        413   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.421E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.204E-03   AT NODE       1422   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.204E-03
 LARGEST INCREMENT OF DISP.         4.422E-05   AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.057E-07   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.607E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.055E-09

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.800E-05,  FRACTION OF STEP COMPLETED  1.365E-06
 STEP TIME COMPLETED       2.730E-04,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.800E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       323.       TIME AVG. FORCE        337.    
 LARGEST SCALED RESIDUAL FORCE     -0.505       AT NODE       1481   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.505    
 LARGEST INCREMENT OF DISP.         2.357E-05   AT NODE         85   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.076E-05   AT NODE        106   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.206E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -5.499E-02   AT NODE       1482   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -5.499E-02
 LARGEST INCREMENT OF DISP.         3.656E-05   AT NODE         85   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.304E-05   AT NODE         79   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.448E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.687E-02   AT NODE        344   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.687E-02
 LARGEST INCREMENT OF DISP.         3.704E-05   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        5.232E-07   AT NODE        106   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    1.605E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.811E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   7.272E-10

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.800E-05,  FRACTION OF STEP COMPLETED  1.605E-06
 STEP TIME COMPLETED       3.210E-04,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.800E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       337.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.472       AT NODE        614   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.472    
 LARGEST INCREMENT OF DISP.         2.598E-05   AT NODE        618   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.177E-05   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -9.890E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -3.657E-02   AT NODE        100   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.657E-02
 LARGEST INCREMENT OF DISP.         3.228E-05   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        6.890E-06   AT NODE         85   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.164E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -2.962E-02   AT NODE         97   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.962E-02
 LARGEST INCREMENT OF DISP.         3.323E-05   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.007E-06   AT NODE        416   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.560E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -3.666E-03   AT NODE       1465   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.666E-03
 LARGEST INCREMENT OF DISP.         3.370E-05   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        4.880E-07   AT NODE         79   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    6.040E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.964E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.996E-10
          TIME INCREMENT MAY NOW INCREASE TO    7.200E-05

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.800E-05,  FRACTION OF STEP COMPLETED  1.845E-06
 STEP TIME COMPLETED       3.690E-04,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.200E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.401       AT NODE       1481   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.401    
 LARGEST INCREMENT OF DISP.         4.227E-05   AT NODE        601   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -8.479E-06   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -2.751E-02   AT NODE        620   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.751E-02
 LARGEST INCREMENT OF DISP.         4.381E-05   AT NODE        106   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.785E-06   AT NODE        107   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION   -1.995E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.268E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.324E-10
          TIME INCREMENT MAY NOW INCREASE TO    1.080E-04

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.200E-05,  FRACTION OF STEP COMPLETED  2.205E-06
 STEP TIME COMPLETED       4.410E-04,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.080E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.293       AT NODE       1490   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.293    
 LARGEST INCREMENT OF DISP.         6.376E-05   AT NODE         85   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.675E-06   AT NODE        586   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.929E-02   AT NODE        352   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.929E-02
 LARGEST INCREMENT OF DISP.         6.500E-05   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.252E-06   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    8.249E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.705E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.549E-10
          TIME INCREMENT MAY NOW INCREASE TO    1.620E-04

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.080E-04,  FRACTION OF STEP COMPLETED  2.745E-06
 STEP TIME COMPLETED       5.490E-04,  TOTAL TIME COMPLETED        0.201    


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.620E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      -1.76       AT NODE         97   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.76    
 LARGEST INCREMENT OF DISP.         7.542E-05   AT NODE         88   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.945E-05   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -4.743E-02   AT NODE       1491   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.743E-02
 LARGEST INCREMENT OF DISP.         8.086E-05   AT NODE        104   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.279E-05   AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.708E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.718E-02   AT NODE        592   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.718E-02
 LARGEST INCREMENT OF DISP.         8.291E-05   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.080E-06   AT NODE        432   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    7.530E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.031E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.126E-09
          TIME INCREMENT MAY NOW INCREASE TO    2.430E-04

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.620E-04,  FRACTION OF STEP COMPLETED  3.555E-06
 STEP TIME COMPLETED       7.110E-04,  TOTAL TIME COMPLETED        0.201    


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.430E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      -1.13       AT NODE        194   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.13    
 LARGEST INCREMENT OF DISP.         8.421E-05   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -4.084E-05   AT NODE        106   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -4.812E-02   AT NODE        349   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.812E-02
 LARGEST INCREMENT OF DISP.         9.921E-05   AT NODE         79   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.504E-05   AT NODE         76   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.946E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.840E-02   AT NODE       1460   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.840E-02
 LARGEST INCREMENT OF DISP.         1.014E-04   AT NODE         79   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.190E-06   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    8.376E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.502E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.987E-10
          TIME INCREMENT MAY NOW INCREASE TO    3.645E-04

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.430E-04,  FRACTION OF STEP COMPLETED  4.770E-06
 STEP TIME COMPLETED       9.540E-04,  TOTAL TIME COMPLETED        0.201    


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.645E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      -1.19       AT NODE         12   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.19    
 LARGEST INCREMENT OF DISP.         1.088E-04   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -4.334E-05   AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -7.870E-02   AT NODE         99   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -7.870E-02
 LARGEST INCREMENT OF DISP.         1.228E-04   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.412E-05   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.816E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      3.745E-02   AT NODE        933   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      3.745E-02
 LARGEST INCREMENT OF DISP.         1.273E-04   AT NODE         79   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        4.465E-06   AT NODE         79   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.751E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.075E-02   AT NODE        194   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.075E-02
 LARGEST INCREMENT OF DISP.         1.294E-04   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.133E-06   AT NODE        107   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION   -6.120E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.117E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.551E-09
          TIME INCREMENT MAY NOW INCREASE TO    5.467E-04

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.645E-04,  FRACTION OF STEP COMPLETED  6.592E-06
 STEP TIME COMPLETED       1.319E-03,  TOTAL TIME COMPLETED        0.201    


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.467E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.926       AT NODE         12   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.926    
 LARGEST INCREMENT OF DISP.         1.385E-04   AT NODE         76   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -5.577E-05   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -7.300E-02   AT NODE        349   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -7.300E-02
 LARGEST INCREMENT OF DISP.         1.484E-04   AT NODE         79   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        9.882E-06   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.479E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -3.102E-02   AT NODE        592   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.102E-02
 LARGEST INCREMENT OF DISP.         1.517E-04   AT NODE         79   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.331E-06   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    1.416E-06
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.025E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   8.669E-10
          TIME INCREMENT MAY NOW INCREASE TO    8.201E-04

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.467E-04,  FRACTION OF STEP COMPLETED  9.326E-06
 STEP TIME COMPLETED       1.865E-03,  TOTAL TIME COMPLETED        0.202    


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.201E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.809       AT NODE         12   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.809    
 LARGEST INCREMENT OF DISP.         1.749E-04   AT NODE         76   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -5.267E-05   AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -9.547E-02   AT NODE        349   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -9.547E-02
 LARGEST INCREMENT OF DISP.         1.812E-04   AT NODE         76   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        6.310E-06   AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    1.080E-06
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.880E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.464E-09
          TIME INCREMENT MAY NOW INCREASE TO    1.230E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.201E-04,  FRACTION OF STEP COMPLETED  1.343E-05
 STEP TIME COMPLETED       2.685E-03,  TOTAL TIME COMPLETED        0.203    


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.230E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.453       AT NODE         12   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.453    
 LARGEST INCREMENT OF DISP.         2.199E-04   AT NODE        416   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -5.231E-05   AT NODE          2   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      8.341E-02   AT NODE        919   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      8.341E-02
 LARGEST INCREMENT OF DISP.         2.236E-04   AT NODE         67   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.816E-06   AT NODE        382   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION   -7.700E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.054E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.281E-09
          TIME INCREMENT MAY NOW INCREASE TO    1.845E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.230E-03,  FRACTION OF STEP COMPLETED  1.958E-05
 STEP TIME COMPLETED       3.916E-03,  TOTAL TIME COMPLETED        0.204    


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.845E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.473       AT NODE         27   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.473    
 LARGEST INCREMENT OF DISP.         2.625E-04   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -7.404E-05   AT NODE        416   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -6.379E-02   AT NODE        628   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -6.379E-02
 LARGEST INCREMENT OF DISP.         2.667E-04   AT NODE         35   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        4.888E-06   AT NODE        416   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    6.588E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.097E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.454E-09
          TIME INCREMENT MAY NOW INCREASE TO    2.768E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.845E-03,  FRACTION OF STEP COMPLETED  2.880E-05
 STEP TIME COMPLETED       5.761E-03,  TOTAL TIME COMPLETED        0.206    


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.768E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  16ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.493       AT NODE         30   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.493    
 LARGEST INCREMENT OF DISP.         3.131E-04   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -8.774E-05   AT NODE         35   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.390E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -5.149E-02   AT NODE        646   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -5.149E-02
 LARGEST INCREMENT OF DISP.         3.140E-04   AT NODE        583   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.321E-06   AT NODE         34   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.461E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.560E-09
          TIME INCREMENT MAY NOW INCREASE TO    4.152E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.768E-03,  FRACTION OF STEP COMPLETED  4.264E-05
 STEP TIME COMPLETED       8.529E-03,  TOTAL TIME COMPLETED        0.209    


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.152E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.171       AT NODE        368   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.171    
 LARGEST INCREMENT OF DISP.         3.716E-04   AT NODE        574   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.001E-04   AT NODE        470   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.868E-02   AT NODE       1514   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.868E-02
 LARGEST INCREMENT OF DISP.         3.722E-04   AT NODE        574   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        6.242E-07   AT NODE         79   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.126E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.812E-09
          TIME INCREMENT MAY NOW INCREASE TO    6.228E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.152E-03,  FRACTION OF STEP COMPLETED  6.340E-05
 STEP TIME COMPLETED       1.268E-02,  TOTAL TIME COMPLETED        0.213    


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.228E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.146       AT NODE        368   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.146    
 LARGEST INCREMENT OF DISP.         4.369E-04   AT NODE        572   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.214E-04   AT NODE        574   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.483E-02   AT NODE         16   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.483E-02
 LARGEST INCREMENT OF DISP.         4.371E-04   AT NODE        572   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.841E-07   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.098E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.035E-09
          TIME INCREMENT MAY NOW INCREASE TO    9.342E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.228E-03,  FRACTION OF STEP COMPLETED  9.454E-05
 STEP TIME COMPLETED       1.891E-02,  TOTAL TIME COMPLETED        0.219    


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  9.342E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.127       AT NODE        359   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.127    
 LARGEST INCREMENT OF DISP.         5.131E-04   AT NODE        573   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.426E-04   AT NODE         62   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -2.374E-02   AT NODE       1514   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.374E-02
 LARGEST INCREMENT OF DISP.         5.135E-04   AT NODE        573   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        4.172E-07   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.840E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.426E-09
          TIME INCREMENT MAY NOW INCREASE TO    1.401E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  9.342E-03,  FRACTION OF STEP COMPLETED  1.413E-04
 STEP TIME COMPLETED       2.825E-02,  TOTAL TIME COMPLETED        0.228    


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.401E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.105       AT NODE        366   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.105    
 LARGEST INCREMENT OF DISP.         6.027E-04   AT NODE        582   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.677E-04   AT NODE        583   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      5.653E-04   AT NODE        370   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.653E-04
 LARGEST INCREMENT OF DISP.         6.029E-04   AT NODE        582   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.076E-07   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.328E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.965E-09
          TIME INCREMENT MAY NOW INCREASE TO    2.102E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.401E-02,  FRACTION OF STEP COMPLETED  2.113E-04
 STEP TIME COMPLETED       4.226E-02,  TOTAL TIME COMPLETED        0.242    


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.102E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -8.272E-02   AT NODE        366   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.272E-02
 LARGEST INCREMENT OF DISP.         7.074E-04   AT NODE        581   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.970E-04   AT NODE        582   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.389E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      5.005E-03   AT NODE        372   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.005E-03
 LARGEST INCREMENT OF DISP.         7.075E-04   AT NODE        581   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.881E-07   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.067E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.705E-09
          TIME INCREMENT MAY NOW INCREASE TO    3.153E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.102E-02,  FRACTION OF STEP COMPLETED  3.164E-04
 STEP TIME COMPLETED       6.328E-02,  TOTAL TIME COMPLETED        0.263    


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.153E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      6.624E-02   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      6.624E-02
 LARGEST INCREMENT OF DISP.         8.300E-04   AT NODE        580   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.312E-04   AT NODE        581   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.390E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -2.024E-03   AT NODE        361   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.024E-03
 LARGEST INCREMENT OF DISP.         8.300E-04   AT NODE        580   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -5.049E-08   AT NODE        109   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.992E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.722E-09
          TIME INCREMENT MAY NOW INCREASE TO    4.729E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.153E-02,  FRACTION OF STEP COMPLETED  4.741E-04
 STEP TIME COMPLETED       9.481E-02,  TOTAL TIME COMPLETED        0.295    


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.729E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.175       AT NODE         25   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.175    
 LARGEST INCREMENT OF DISP.         9.731E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.720E-04   AT NODE        580   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.390E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      2.841E-02   AT NODE        368   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.841E-02
 LARGEST INCREMENT OF DISP.         9.736E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        5.810E-07   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.344E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.123E-09
          TIME INCREMENT MAY NOW INCREASE TO    7.094E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.729E-02,  FRACTION OF STEP COMPLETED  7.105E-04
 STEP TIME COMPLETED       0.142    ,  TOTAL TIME COMPLETED        0.342    


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.094E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.133       AT NODE         25   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.133    
 LARGEST INCREMENT OF DISP.         1.142E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.188E-04   AT NODE        577   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.390E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -3.526E-03   AT NODE        361   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.526E-03
 LARGEST INCREMENT OF DISP.         1.142E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.650E-07   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.079E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   7.084E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.106    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.094E-02,  FRACTION OF STEP COMPLETED  1.065E-03
 STEP TIME COMPLETED       0.213    ,  TOTAL TIME COMPLETED        0.413    


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.106    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.303       AT NODE         26   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.303    
 LARGEST INCREMENT OF DISP.         1.339E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.742E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.390E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.137E-02   AT NODE        363   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.137E-02
 LARGEST INCREMENT OF DISP.         1.339E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        5.128E-07   AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.278E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   9.732E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.160    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.106    ,  FRACTION OF STEP COMPLETED  1.597E-03
 STEP TIME COMPLETED       0.319    ,  TOTAL TIME COMPLETED        0.519    


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.160    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       339.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.283       AT NODE         26   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.283    
 LARGEST INCREMENT OF DISP.         1.570E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -4.384E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.391E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       339.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      1.716E-02   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.716E-02
 LARGEST INCREMENT OF DISP.         1.571E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        4.526E-07   AT NODE        388   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.172E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.336E-08
          TIME INCREMENT MAY NOW INCREASE TO    0.239    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.160    ,  FRACTION OF STEP COMPLETED  2.395E-03
 STEP TIME COMPLETED       0.479    ,  TOTAL TIME COMPLETED        0.679    


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.239    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       339.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.420       AT NODE         27   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.420    
 LARGEST INCREMENT OF DISP.         1.841E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.146E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.391E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       339.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      4.093E-02   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      4.093E-02
 LARGEST INCREMENT OF DISP.         1.842E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        6.725E-07   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.878E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.591E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.836E-08
          TIME INCREMENT MAY NOW INCREASE TO    0.359    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.239    ,  FRACTION OF STEP COMPLETED  3.592E-03
 STEP TIME COMPLETED       0.718    ,  TOTAL TIME COMPLETED        0.918    


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.359    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       339.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.441       AT NODE         27   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.441    
 LARGEST INCREMENT OF DISP.         2.159E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -6.033E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.879E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.391E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       339.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -1.196E-02   AT NODE        361   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.196E-02
 LARGEST INCREMENT OF DISP.         2.160E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        8.511E-07   AT NODE         27   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.879E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.708E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.519E-08
          TIME INCREMENT MAY NOW INCREASE TO    0.539    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.359    ,  FRACTION OF STEP COMPLETED  5.388E-03
 STEP TIME COMPLETED        1.08    ,  TOTAL TIME COMPLETED         1.28    


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.539    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       339.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.464       AT NODE         27   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.464    
 LARGEST INCREMENT OF DISP.         2.533E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -7.078E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.879E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.391E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       339.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -3.182E-02   AT NODE        363   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.182E-02
 LARGEST INCREMENT OF DISP.         2.533E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        7.200E-07   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.879E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.118E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.458E-08
          TIME INCREMENT MAY NOW INCREASE TO    0.808    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.539    ,  FRACTION OF STEP COMPLETED  8.081E-03
 STEP TIME COMPLETED        1.62    ,  TOTAL TIME COMPLETED         1.82    


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.808    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       339.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.492       AT NODE         27   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.492    
 LARGEST INCREMENT OF DISP.         2.971E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -8.289E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.879E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.392E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       339.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -8.969E-03   AT NODE        361   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.969E-03
 LARGEST INCREMENT OF DISP.         2.971E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        5.723E-07   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.879E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.432E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.751E-08
          TIME INCREMENT MAY NOW INCREASE TO     1.21    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.808    ,  FRACTION OF STEP COMPLETED  1.212E-02
 STEP TIME COMPLETED        2.42    ,  TOTAL TIME COMPLETED         2.62    


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   1.21    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       339.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.531       AT NODE         27   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.531    
 LARGEST INCREMENT OF DISP.         3.485E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -9.718E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.879E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.392E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       339.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -4.419E-02   AT NODE        363   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.419E-02
 LARGEST INCREMENT OF DISP.         3.485E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -8.231E-07   AT NODE        264   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.879E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.180E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   6.522E-08
          TIME INCREMENT MAY NOW INCREASE TO     1.82    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   1.21    ,  FRACTION OF STEP COMPLETED  1.818E-02
 STEP TIME COMPLETED        3.64    ,  TOTAL TIME COMPLETED         3.84    


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   1.82    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       339.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.564       AT NODE         27   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.564    
 LARGEST INCREMENT OF DISP.         4.110E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.117E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.879E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.393E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       340.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -2.624E-04   AT NODE        400   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.624E-04
 LARGEST INCREMENT OF DISP.         4.088E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.313E-05   AT NODE        270   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.879E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.178E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   8.944E-08
          TIME INCREMENT MAY NOW INCREASE TO     2.73    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   1.82    ,  FRACTION OF STEP COMPLETED  2.727E-02
 STEP TIME COMPLETED        5.45    ,  TOTAL TIME COMPLETED         5.65    


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   2.73    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       340.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.834       AT NODE        373   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.834    
 LARGEST INCREMENT OF DISP.         4.795E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.337E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.879E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.394E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       340.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -7.626E-02   AT NODE        364   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -7.626E-02
 LARGEST INCREMENT OF DISP.         4.794E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.126E-06   AT NODE          5   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.879E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.726E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.225E-07
          TIME INCREMENT MAY NOW INCREASE TO     4.09    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   2.73    ,  FRACTION OF STEP COMPLETED  4.091E-02
 STEP TIME COMPLETED        8.18    ,  TOTAL TIME COMPLETED         8.38    


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   4.09    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  10ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       340.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.918       AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.918    
 LARGEST INCREMENT OF DISP.         5.599E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.593E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.880E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.394E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       340.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      1.768E-02   AT NODE        376   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.768E-02
 LARGEST INCREMENT OF DISP.         5.600E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.125E-06   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.880E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.139E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.676E-07
          TIME INCREMENT MAY NOW INCREASE TO     6.14    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   4.09    ,  FRACTION OF STEP COMPLETED  6.136E-02
 STEP TIME COMPLETED        12.3    ,  TOTAL TIME COMPLETED         12.5    


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.14    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       340.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      -1.01       AT NODE        373   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.01    
 LARGEST INCREMENT OF DISP.         6.568E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.832E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.880E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.395E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       340.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE     -0.106       AT NODE        364   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.106    
 LARGEST INCREMENT OF DISP.         6.569E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.428E-06   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.880E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.549E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.305E-07
          TIME INCREMENT MAY NOW INCREASE TO     9.20    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.14    ,  FRACTION OF STEP COMPLETED  9.204E-02
 STEP TIME COMPLETED        18.4    ,  TOTAL TIME COMPLETED         18.6    


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   9.20    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       340.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      -1.12       AT NODE        373   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.12    
 LARGEST INCREMENT OF DISP.         7.711E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.143E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.880E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.396E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       340.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.134       AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.134    
 LARGEST INCREMENT OF DISP.         7.712E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.497E-06   AT NODE        391   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.880E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.869E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.175E-07
          TIME INCREMENT MAY NOW INCREASE TO     13.8    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   9.20    ,  FRACTION OF STEP COMPLETED  0.138    
 STEP TIME COMPLETED        27.6    ,  TOTAL TIME COMPLETED         27.8    


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   13.8    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      -1.27       AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.27    
 LARGEST INCREMENT OF DISP.         9.054E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.514E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.881E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.397E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       338.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.169       AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.169    
 LARGEST INCREMENT OF DISP.         9.055E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.334E-06   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.881E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.113E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.375E-07
          TIME INCREMENT MAY NOW INCREASE TO     20.7    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   13.8    ,  FRACTION OF STEP COMPLETED  0.207    
 STEP TIME COMPLETED        41.4    ,  TOTAL TIME COMPLETED         41.6    


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   20.7    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       336.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      -1.44       AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.44    
 LARGEST INCREMENT OF DISP.         1.063E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.948E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.881E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.399E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       336.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      7.074E-02   AT NODE         26   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      7.074E-02
 LARGEST INCREMENT OF DISP.         1.064E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.838E-06   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.881E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.269E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   6.033E-07
          TIME INCREMENT MAY NOW INCREASE TO     31.1    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   20.7    ,  FRACTION OF STEP COMPLETED  0.311    
 STEP TIME COMPLETED        62.1    ,  TOTAL TIME COMPLETED         62.3    


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   31.1    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       333.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE       1.58       AT NODE        384   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       1.58    
 LARGEST INCREMENT OF DISP.         1.249E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.460E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.882E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.400E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       334.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.130       AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.130    
 LARGEST INCREMENT OF DISP.         1.249E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.542E-06   AT NODE          5   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.882E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.378E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   8.326E-07
          TIME INCREMENT MAY NOW INCREASE TO     46.6    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   31.1    ,  FRACTION OF STEP COMPLETED  0.466    
 STEP TIME COMPLETED        93.2    ,  TOTAL TIME COMPLETED         93.4    


  INCREMENT    49 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   46.6    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       330.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      -1.87       AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.87    
 LARGEST INCREMENT OF DISP.         1.468E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -4.058E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.882E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.402E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       331.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      0.226       AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.226    
 LARGEST INCREMENT OF DISP.         1.468E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.944E-06   AT NODE        111   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.882E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.624E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.149E-06
          TIME INCREMENT MAY NOW INCREASE TO     60.2    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   46.6    ,  FRACTION OF STEP COMPLETED  0.699    
 STEP TIME COMPLETED        140.    ,  TOTAL TIME COMPLETED         140.    


  INCREMENT    50 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   60.2    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       328.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      -2.09       AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -2.09    
 LARGEST INCREMENT OF DISP.         1.510E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.872E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.883E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.495E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       328.       TIME AVG. FORCE        338.    
 LARGEST SCALED RESIDUAL FORCE      8.577E-02   AT NODE         26   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      8.577E-02
 LARGEST INCREMENT OF DISP.         1.510E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.558E-06   AT NODE          5   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       157.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.883E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.057E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.231E-06

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   60.2    ,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED        200.    ,  TOTAL TIME COMPLETED         200.    


          THE ANALYSIS HAS BEEN COMPLETED



     ANALYSIS SUMMARY:
     TOTAL OF         51  INCREMENTS
                       0  CUTBACKS IN AUTOMATIC INCREMENTATION
                     126  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     126  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                     126  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       2  WARNING MESSAGES DURING USER INPUT PROCESSING
                       4  WARNING MESSAGES DURING ANALYSIS
                       0  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                       0  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =      2.3    
       SYSTEM TIME (SEC)    =     0.30    
       TOTAL CPU TIME (SEC) =      2.6    
       WALLCLOCK TIME (SEC) =            9
       MEMORY PEAK (GB)     =            0
