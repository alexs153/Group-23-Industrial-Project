1

   Abaqus 2024                                  Date 20-Mar-2025   Time 15:16:25
   For use by UNIVERSITY OF BRISTOL under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     S T A T I C   A N A L Y S I S


                                                                                          

     AUTOMATIC TIME CONTROL WITH -
          A SUGGESTED INITIAL TIME INCREMENT OF                0.140    
          AND A TOTAL TIME PERIOD OF                           0.140    
          THE MINIMUM TIME INCREMENT ALLOWED IS                1.400E-06
          THE MAXIMUM TIME INCREMENT ALLOWED IS                0.140    

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


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.140    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       173.       TIME AVG. FORCE        173.    
 LARGEST SCALED RESIDUAL FORCE      -658.       AT NODE         96   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -658.    
 LARGEST INCREMENT OF DISP.         8.287E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        8.287E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       80.4    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.642E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.642E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       162.       TIME AVG. FORCE        162.    
 LARGEST SCALED RESIDUAL FORCE     -1.117E+03   AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.117E+03
 LARGEST INCREMENT OF DISP.          1.03       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        0.949       AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       75.2    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.531E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 685 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       222.       TIME AVG. FORCE        222.    
 LARGEST SCALED RESIDUAL FORCE      1.639E+03   AT NODE        196   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.639E+03
 LARGEST INCREMENT OF DISP.          2.68       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.         1.65       AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       103.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.201E-36   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 689 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       190.       TIME AVG. FORCE        190.    
 LARGEST SCALED RESIDUAL FORCE       56.9       AT NODE        344   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       56.9    
 LARGEST INCREMENT OF DISP.          2.90       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        0.220       AT NODE        103   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       88.3    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.283E-36   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 690 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       190.       TIME AVG. FORCE        190.    
 LARGEST SCALED RESIDUAL FORCE      -2.35       AT NODE        344   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -2.35    
 LARGEST INCREMENT OF DISP.          2.92       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.998E-02   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       88.3    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.008E-37   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 690 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                       190.       TIME AVG. FORCE        190.    
 LARGEST SCALED RESIDUAL FORCE     -2.619E-02   AT NODE         97   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.619E-02
 LARGEST INCREMENT OF DISP.          2.92       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.876E-05   AT NODE        109   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       88.3    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.439E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.140    ,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED       0.140    ,  TOTAL TIME COMPLETED        0.140    
1

   Abaqus 2024                                  Date 20-Mar-2025   Time 15:16:26
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
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -6.514E-03   AT NODE         98   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -6.514E-03
 LARGEST INCREMENT OF DISP.        -6.917E-08   AT NODE         98   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -6.917E-08   AT NODE         98   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      2.138E-03   AT NODE        615   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.138E-03
 LARGEST INCREMENT OF DISP.        -5.509E-08   AT NODE         98   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.371E-08   AT NODE         99   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.275E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -1.376E-03   AT NODE        597   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.376E-03
 LARGEST INCREMENT OF DISP.        -3.591E-08   AT NODE         98   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.163E-08   AT NODE        623   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.264E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  10ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -7.206E-04   AT NODE       1514   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -7.206E-04
 LARGEST INCREMENT OF DISP.         3.946E-08   AT NODE       1450   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.388E-08   AT NODE        104   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.039E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -7.244E-04   AT NODE        632   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -7.244E-04
 LARGEST INCREMENT OF DISP.         4.167E-08   AT NODE       1450   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        4.752E-09   AT NODE        339   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.749E-43   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -6.241E-04   AT NODE        623   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -6.241E-04
 LARGEST INCREMENT OF DISP.         4.221E-08   AT NODE       1450   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.252E-09   AT NODE        640   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.642E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -4.823E-04   AT NODE        630   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.823E-04
 LARGEST INCREMENT OF DISP.         4.283E-08   AT NODE       1450   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.713E-09   AT NODE        631   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.216E-43   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     8


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      3.796E-04   AT NODE        637   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      3.796E-04
 LARGEST INCREMENT OF DISP.         4.313E-08   AT NODE       1450   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.050E-09   AT NODE        638   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.768E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  10ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     9


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      5.280E-05   AT NODE       1433   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.280E-05
 LARGEST INCREMENT OF DISP.         4.313E-08   AT NODE       1450   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.055E-09   AT NODE        637   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    1.468E-10
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.050E-43   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.287E-15

 ITERATION SUMMARY FOR THE INCREMENT:   9 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  9 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-06,  FRACTION OF STEP COMPLETED  5.000E-09
 STEP TIME COMPLETED       1.000E-06,  TOTAL TIME COMPLETED        0.140    


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -6.130E-04   AT NODE       1458   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -6.130E-04
 LARGEST INCREMENT OF DISP.         9.896E-09   AT NODE        643   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -4.179E-08   AT NODE       1450   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.709E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      8.802E-05   AT NODE       1510   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      8.802E-05
 LARGEST INCREMENT OF DISP.         7.818E-09   AT NODE        104   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.122E-09   AT NODE        636   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.364E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      4.640E-08   AT NODE       1502   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      4.640E-08
 LARGEST INCREMENT OF DISP.         7.802E-09   AT NODE        104   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.034E-10   AT NODE        339   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.454E-43   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   6.709E-16

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-06,  FRACTION OF STEP COMPLETED  1.000E-08
 STEP TIME COMPLETED       2.000E-06,  TOTAL TIME COMPLETED        0.140    


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      1.050E-06   AT NODE        352   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.050E-06
 LARGEST INCREMENT OF DISP.         7.057E-09   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.018E-09   AT NODE       1385   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.102E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.908E-18
          TIME INCREMENT MAY NOW INCREASE TO    2.000E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-06,  FRACTION OF STEP COMPLETED  1.500E-08
 STEP TIME COMPLETED       3.000E-06,  TOTAL TIME COMPLETED        0.140    


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -8.009E-10   AT NODE       1386   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.009E-10
 LARGEST INCREMENT OF DISP.         1.411E-08   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.852E-12   AT NODE        273   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   7.792E-18
          TIME INCREMENT MAY NOW INCREASE TO    4.000E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.000E-06,  FRACTION OF STEP COMPLETED  2.500E-08
 STEP TIME COMPLETED       5.000E-06,  TOTAL TIME COMPLETED        0.140    


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      5.350E-10   AT NODE       1386   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.350E-10
 LARGEST INCREMENT OF DISP.         2.822E-08   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.028E-12   AT NODE        352   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.096E-17
          TIME INCREMENT MAY NOW INCREASE TO    8.000E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.000E-06,  FRACTION OF STEP COMPLETED  4.500E-08
 STEP TIME COMPLETED       9.000E-06,  TOTAL TIME COMPLETED        0.140    


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      4.002E-11   AT NODE          1   DOF  2
   INSTANCE:                                                                                 
  CORRESPONDING RESIDUAL FORCE      4.002E-11
 LARGEST INCREMENT OF DISP.         5.645E-08   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.470E-12   AT NODE         59   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.238E-16
          TIME INCREMENT MAY NOW INCREASE TO    1.600E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.000E-06,  FRACTION OF STEP COMPLETED  8.500E-08
 STEP TIME COMPLETED       1.700E-05,  TOTAL TIME COMPLETED        0.140    


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.600E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      3.092E-11   AT NODE          1   DOF  2
   INSTANCE:                                                                                 
  CORRESPONDING RESIDUAL FORCE      3.092E-11
 LARGEST INCREMENT OF DISP.         1.129E-07   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.880E-12   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.953E-16
          TIME INCREMENT MAY NOW INCREASE TO    3.200E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.600E-05,  FRACTION OF STEP COMPLETED  1.650E-07
 STEP TIME COMPLETED       3.300E-05,  TOTAL TIME COMPLETED        0.140    


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.200E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -3.160E-11   AT NODE        440   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.160E-11
 LARGEST INCREMENT OF DISP.         2.258E-07   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.352E-11   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.981E-15
          TIME INCREMENT MAY NOW INCREASE TO    6.400E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.200E-05,  FRACTION OF STEP COMPLETED  3.250E-07
 STEP TIME COMPLETED       6.500E-05,  TOTAL TIME COMPLETED        0.140    


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.400E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.528       AT NODE        262   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.528    
 LARGEST INCREMENT OF DISP.         1.131E-04   AT NODE        264   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.126E-04   AT NODE        264   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.156       AT NODE        190   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.156    
 LARGEST INCREMENT OF DISP.         1.198E-04   AT NODE        111   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        7.421E-06   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -9.540E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      5.136E-02   AT NODE       1462   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.136E-02
 LARGEST INCREMENT OF DISP.         1.280E-04   AT NODE        109   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        8.591E-06   AT NODE        104   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.886E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -1.621E-02   AT NODE        602   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.621E-02
 LARGEST INCREMENT OF DISP.         1.336E-04   AT NODE        109   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        5.641E-06   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.676E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -6.709E-03   AT NODE        621   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -6.709E-03
 LARGEST INCREMENT OF DISP.         1.343E-04   AT NODE        609   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        7.684E-07   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.145E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   9.591E-10

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.400E-05,  FRACTION OF STEP COMPLETED  6.450E-07
 STEP TIME COMPLETED       1.290E-04,  TOTAL TIME COMPLETED        0.140    


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.400E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.218       AT NODE        196   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.218    
 LARGEST INCREMENT OF DISP.         5.907E-05   AT NODE        106   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -7.693E-05   AT NODE        264   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.546E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -1.049E-02   AT NODE        626   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.049E-02
 LARGEST INCREMENT OF DISP.         5.995E-05   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.011E-06   AT NODE        334   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION   -4.877E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.697E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.742E-09

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.400E-05,  FRACTION OF STEP COMPLETED  9.650E-07
 STEP TIME COMPLETED       1.930E-04,  TOTAL TIME COMPLETED        0.140    


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.400E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      -1.26       AT NODE         97   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.26    
 LARGEST INCREMENT OF DISP.         2.044E-05   AT NODE        270   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -4.339E-05   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.019E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.115       AT NODE       1466   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.115    
 LARGEST INCREMENT OF DISP.         3.153E-05   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.495E-05   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.434E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -7.118E-02   AT NODE        351   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -7.118E-02
 LARGEST INCREMENT OF DISP.         3.369E-05   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.220E-06   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.258E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      2.229E-02   AT NODE        645   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.229E-02
 LARGEST INCREMENT OF DISP.         3.496E-05   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.338E-06   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.727E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -5.469E-03   AT NODE         15   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -5.469E-03
 LARGEST INCREMENT OF DISP.         3.532E-05   AT NODE         85   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.976E-07   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION   -9.755E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.603E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   8.248E-10

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.400E-05,  FRACTION OF STEP COMPLETED  1.285E-06
 STEP TIME COMPLETED       2.570E-04,  TOTAL TIME COMPLETED        0.140    


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.400E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.715       AT NODE        195   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.715    
 LARGEST INCREMENT OF DISP.         1.794E-05   AT NODE       1393   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.888E-05   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.275E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      6.126E-02   AT NODE       1474   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      6.126E-02
 LARGEST INCREMENT OF DISP.         2.573E-05   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        9.130E-06   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.758E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -1.643E-02   AT NODE       1491   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.643E-02
 LARGEST INCREMENT OF DISP.         2.668E-05   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        9.753E-07   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION   -2.616E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.682E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   6.935E-10

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.400E-05,  FRACTION OF STEP COMPLETED  1.605E-06
 STEP TIME COMPLETED       3.210E-04,  TOTAL TIME COMPLETED        0.140    


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.400E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.446       AT NODE         98   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.446    
 LARGEST INCREMENT OF DISP.         2.028E-05   AT NODE        641   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -6.537E-06   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.117E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      4.876E-02   AT NODE       1475   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      4.876E-02
 LARGEST INCREMENT OF DISP.         2.249E-05   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.378E-06   AT NODE        108   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.907E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -1.896E-02   AT NODE        347   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.896E-02
 LARGEST INCREMENT OF DISP.         2.369E-05   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.196E-06   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.301E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      4.216E-03   AT NODE        883   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      4.216E-03
 LARGEST INCREMENT OF DISP.         2.406E-05   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.909E-07   AT NODE         78   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION   -8.694E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.189E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.995E-10
          TIME INCREMENT MAY NOW INCREASE TO    9.600E-05

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.400E-05,  FRACTION OF STEP COMPLETED  1.925E-06
 STEP TIME COMPLETED       3.850E-04,  TOTAL TIME COMPLETED        0.140    


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  9.600E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.511       AT NODE         98   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.511    
 LARGEST INCREMENT OF DISP.         2.568E-05   AT NODE        111   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.110E-05   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -4.109E-02   AT NODE        613   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.109E-02
 LARGEST INCREMENT OF DISP.         2.836E-05   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.291E-06   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.374E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -1.362E-02   AT NODE         96   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.362E-02
 LARGEST INCREMENT OF DISP.         2.915E-05   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        7.828E-07   AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    2.594E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.983E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.844E-10
          TIME INCREMENT MAY NOW INCREASE TO    1.440E-04

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  9.600E-05,  FRACTION OF STEP COMPLETED  2.405E-06
 STEP TIME COMPLETED       4.810E-04,  TOTAL TIME COMPLETED        0.140    


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.440E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.582       AT NODE         99   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.582    
 LARGEST INCREMENT OF DISP.         3.346E-05   AT NODE          5   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.086E-05   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -3.111E-02   AT NODE       1483   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.111E-02
 LARGEST INCREMENT OF DISP.         3.591E-05   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.998E-06   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.889E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -1.195E-02   AT NODE         97   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.195E-02
 LARGEST INCREMENT OF DISP.         3.712E-05   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.209E-06   AT NODE         74   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.590E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -2.959E-03   AT NODE        189   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.959E-03
 LARGEST INCREMENT OF DISP.         3.736E-05   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.655E-07   AT NODE         79   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.029E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.526E-10
          TIME INCREMENT MAY NOW INCREASE TO    2.160E-04

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.440E-04,  FRACTION OF STEP COMPLETED  3.125E-06
 STEP TIME COMPLETED       6.250E-04,  TOTAL TIME COMPLETED        0.141    


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.160E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.729       AT NODE         99   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.729    
 LARGEST INCREMENT OF DISP.         3.548E-05   AT NODE        265   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.184E-05   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  11ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -4.224E-02   AT NODE        621   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.224E-02
 LARGEST INCREMENT OF DISP.         4.358E-05   AT NODE        632   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        9.346E-06   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.003E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -1.819E-02   AT NODE         97   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.819E-02
 LARGEST INCREMENT OF DISP.         4.443E-05   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        8.525E-07   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    3.671E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.647E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.277E-10
          TIME INCREMENT MAY NOW INCREASE TO    3.240E-04

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.160E-04,  FRACTION OF STEP COMPLETED  4.205E-06
 STEP TIME COMPLETED       8.410E-04,  TOTAL TIME COMPLETED        0.141    


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.240E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.696       AT NODE        100   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.696    
 LARGEST INCREMENT OF DISP.         4.698E-05   AT NODE        264   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.058E-05   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -4.346E-02   AT NODE         97   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.346E-02
 LARGEST INCREMENT OF DISP.         5.368E-05   AT NODE         85   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        7.560E-06   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.419E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -1.620E-02   AT NODE       1406   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.620E-02
 LARGEST INCREMENT OF DISP.         5.551E-05   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.846E-06   AT NODE         77   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.730E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -4.048E-03   AT NODE       1412   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.048E-03
 LARGEST INCREMENT OF DISP.         5.651E-05   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.021E-06   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    2.551E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.786E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.145E-10
          TIME INCREMENT MAY NOW INCREASE TO    4.860E-04

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.240E-04,  FRACTION OF STEP COMPLETED  5.825E-06
 STEP TIME COMPLETED       1.165E-03,  TOTAL TIME COMPLETED        0.141    


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.860E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.658       AT NODE        347   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.658    
 LARGEST INCREMENT OF DISP.         5.610E-05   AT NODE         88   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.905E-05   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -5.619E-02   AT NODE         98   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -5.619E-02
 LARGEST INCREMENT OF DISP.         6.452E-05   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        8.773E-06   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.692E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  10ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -1.364E-02   AT NODE        597   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.364E-02
 LARGEST INCREMENT OF DISP.         6.571E-05   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.194E-06   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    2.898E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.512E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.196E-10
          TIME INCREMENT MAY NOW INCREASE TO    7.290E-04

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.860E-04,  FRACTION OF STEP COMPLETED  8.255E-06
 STEP TIME COMPLETED       1.651E-03,  TOTAL TIME COMPLETED        0.142    


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.290E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.612       AT NODE         96   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.612    
 LARGEST INCREMENT OF DISP.         7.243E-05   AT NODE        110   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.641E-05   AT NODE        643   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  10ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -6.397E-02   AT NODE         98   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -6.397E-02
 LARGEST INCREMENT OF DISP.         7.795E-05   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        5.730E-06   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.061E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -1.499E-02   AT NODE       1412   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.499E-02
 LARGEST INCREMENT OF DISP.         8.057E-05   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.636E-06   AT NODE        624   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    6.176E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.240E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.201E-10
          TIME INCREMENT MAY NOW INCREASE TO    1.093E-03

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.290E-04,  FRACTION OF STEP COMPLETED  1.190E-05
 STEP TIME COMPLETED       2.380E-03,  TOTAL TIME COMPLETED        0.142    


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.093E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  10ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.631       AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.631    
 LARGEST INCREMENT OF DISP.         8.744E-05   AT NODE        109   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -3.371E-05   AT NODE         20   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -5.976E-02   AT NODE         98   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -5.976E-02
 LARGEST INCREMENT OF DISP.         9.258E-05   AT NODE         78   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        5.304E-06   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    7.043E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.822E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   7.315E-10
          TIME INCREMENT MAY NOW INCREASE TO    1.640E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.093E-03,  FRACTION OF STEP COMPLETED  1.737E-05
 STEP TIME COMPLETED       3.473E-03,  TOTAL TIME COMPLETED        0.143    


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.640E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.435       AT NODE        101   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.435    
 LARGEST INCREMENT OF DISP.         1.095E-04   AT NODE        107   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.982E-05   AT NODE        384   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -8.009E-02   AT NODE         98   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.009E-02
 LARGEST INCREMENT OF DISP.         1.126E-04   AT NODE         30   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.338E-06   AT NODE         78   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    6.141E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.953E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.320E-10
          TIME INCREMENT MAY NOW INCREASE TO    2.460E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.640E-03,  FRACTION OF STEP COMPLETED  2.557E-05
 STEP TIME COMPLETED       5.114E-03,  TOTAL TIME COMPLETED        0.145    


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.460E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.422       AT NODE         25   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.422    
 LARGEST INCREMENT OF DISP.         1.318E-04   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -3.760E-05   AT NODE         30   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -4.403E-02   AT NODE         12   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.403E-02
 LARGEST INCREMENT OF DISP.         1.343E-04   AT NODE         32   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.806E-06   AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    3.352E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.012E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.068E-09
          TIME INCREMENT MAY NOW INCREASE TO    3.691E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.460E-03,  FRACTION OF STEP COMPLETED  3.787E-05
 STEP TIME COMPLETED       7.574E-03,  TOTAL TIME COMPLETED        0.148    


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.691E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.436       AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.436    
 LARGEST INCREMENT OF DISP.         1.561E-04   AT NODE        583   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -4.540E-05   AT NODE         32   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -3.550E-02   AT NODE       1489   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.550E-02
 LARGEST INCREMENT OF DISP.         1.575E-04   AT NODE        583   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.411E-06   AT NODE        380   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.450E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.832E-09
          TIME INCREMENT MAY NOW INCREASE TO    5.536E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.691E-03,  FRACTION OF STEP COMPLETED  5.632E-05
 STEP TIME COMPLETED       1.126E-02,  TOTAL TIME COMPLETED        0.151    


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.536E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.422       AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.422    
 LARGEST INCREMENT OF DISP.         1.853E-04   AT NODE        564   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -5.149E-05   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      1.076E-02   AT NODE        646   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.076E-02
 LARGEST INCREMENT OF DISP.         1.861E-04   AT NODE        581   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        8.018E-07   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.083E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.984E-09
          TIME INCREMENT MAY NOW INCREASE TO    8.304E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.536E-03,  FRACTION OF STEP COMPLETED  8.400E-05
 STEP TIME COMPLETED       1.680E-02,  TOTAL TIME COMPLETED        0.157    


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.304E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.458       AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.458    
 LARGEST INCREMENT OF DISP.         2.182E-04   AT NODE         60   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -6.102E-05   AT NODE        557   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      1.578E-02   AT NODE       1487   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.578E-02
 LARGEST INCREMENT OF DISP.         2.187E-04   AT NODE         60   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        5.125E-07   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.428E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.873E-09
          TIME INCREMENT MAY NOW INCREASE TO    1.246E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.304E-03,  FRACTION OF STEP COMPLETED  1.255E-04
 STEP TIME COMPLETED       2.510E-02,  TOTAL TIME COMPLETED        0.165    


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.246E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.171       AT NODE         19   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.171    
 LARGEST INCREMENT OF DISP.         2.567E-04   AT NODE         60   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -7.137E-05   AT NODE         62   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -1.255E-02   AT NODE        344   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.255E-02
 LARGEST INCREMENT OF DISP.         2.570E-04   AT NODE         60   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.711E-07   AT NODE        359   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.749E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.884E-10
          TIME INCREMENT MAY NOW INCREASE TO    1.868E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.246E-02,  FRACTION OF STEP COMPLETED  1.878E-04
 STEP TIME COMPLETED       3.756E-02,  TOTAL TIME COMPLETED        0.178    


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.868E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.331       AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.331    
 LARGEST INCREMENT OF DISP.         3.012E-04   AT NODE        582   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -8.429E-05   AT NODE        574   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -5.084E-03   AT NODE       1499   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -5.084E-03
 LARGEST INCREMENT OF DISP.         3.015E-04   AT NODE        582   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.793E-07   AT NODE        359   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.837E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.344E-10
          TIME INCREMENT MAY NOW INCREASE TO    2.803E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.868E-02,  FRACTION OF STEP COMPLETED  2.812E-04
 STEP TIME COMPLETED       5.624E-02,  TOTAL TIME COMPLETED        0.196    


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.803E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.247       AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.247    
 LARGEST INCREMENT OF DISP.         3.532E-04   AT NODE        577   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -9.910E-05   AT NODE        582   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -2.119E-02   AT NODE        640   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.119E-02
 LARGEST INCREMENT OF DISP.         3.536E-04   AT NODE        577   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        4.431E-07   AT NODE        103   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.645E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.636E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   7.351E-10
          TIME INCREMENT MAY NOW INCREASE TO    4.204E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.803E-02,  FRACTION OF STEP COMPLETED  4.213E-04
 STEP TIME COMPLETED       8.427E-02,  TOTAL TIME COMPLETED        0.224    


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.204E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.186       AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.186    
 LARGEST INCREMENT OF DISP.         4.143E-04   AT NODE         59   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.161E-04   AT NODE         60   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -2.194E-02   AT NODE       1499   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.194E-02
 LARGEST INCREMENT OF DISP.         4.147E-04   AT NODE         59   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        4.419E-07   AT NODE        359   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.537E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.012E-09
          TIME INCREMENT MAY NOW INCREASE TO    6.306E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.204E-02,  FRACTION OF STEP COMPLETED  6.315E-04
 STEP TIME COMPLETED       0.126    ,  TOTAL TIME COMPLETED        0.266    


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.306E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.111       AT NODE       1500   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.111    
 LARGEST INCREMENT OF DISP.         4.860E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.360E-04   AT NODE        580   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       294.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -8.889E-03   AT NODE        648   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.889E-03
 LARGEST INCREMENT OF DISP.         4.863E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.126E-07   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.329E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.390E-09
          TIME INCREMENT MAY NOW INCREASE TO    9.459E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.306E-02,  FRACTION OF STEP COMPLETED  9.468E-04
 STEP TIME COMPLETED       0.189    ,  TOTAL TIME COMPLETED        0.329    


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  9.459E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.199       AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.199    
 LARGEST INCREMENT OF DISP.         5.701E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.594E-04   AT NODE        579   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -1.753E-02   AT NODE        648   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.753E-02
 LARGEST INCREMENT OF DISP.         5.703E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.814E-07   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -8.233E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.912E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.142    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  9.459E-02,  FRACTION OF STEP COMPLETED  1.420E-03
 STEP TIME COMPLETED       0.284    ,  TOTAL TIME COMPLETED        0.424    


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.142    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.212       AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.212    
 LARGEST INCREMENT OF DISP.         6.686E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.869E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.227E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -5.289E-05   AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -5.289E-05
 LARGEST INCREMENT OF DISP.         6.688E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.832E-07   AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.559E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.630E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.213    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.142    ,  FRACTION OF STEP COMPLETED  2.129E-03
 STEP TIME COMPLETED       0.426    ,  TOTAL TIME COMPLETED        0.566    


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.213    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.226       AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.226    
 LARGEST INCREMENT OF DISP.         7.840E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.192E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.228E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      1.846E-03   AT NODE        373   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.846E-03
 LARGEST INCREMENT OF DISP.         7.842E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.205E-07   AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.154E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.615E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.319    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.213    ,  FRACTION OF STEP COMPLETED  3.193E-03
 STEP TIME COMPLETED       0.639    ,  TOTAL TIME COMPLETED        0.779    


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.319    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.451       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.451    
 LARGEST INCREMENT OF DISP.         9.191E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.572E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.228E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      1.176E-02   AT NODE        359   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.176E-02
 LARGEST INCREMENT OF DISP.         9.196E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        5.643E-07   AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.341E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.972E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.479    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.319    ,  FRACTION OF STEP COMPLETED  4.789E-03
 STEP TIME COMPLETED       0.958    ,  TOTAL TIME COMPLETED         1.10    


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.479    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.482       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.482    
 LARGEST INCREMENT OF DISP.         1.078E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.017E-04   AT NODE         59   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.228E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -2.756E-02   AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.756E-02
 LARGEST INCREMENT OF DISP.         1.078E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        7.390E-07   AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.472E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   6.838E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.718    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.479    ,  FRACTION OF STEP COMPLETED  7.184E-03
 STEP TIME COMPLETED        1.44    ,  TOTAL TIME COMPLETED         1.58    


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.718    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.512       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.512    
 LARGEST INCREMENT OF DISP.         1.264E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.535E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.228E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      2.278E-02   AT NODE        359   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.278E-02
 LARGEST INCREMENT OF DISP.         1.264E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        6.448E-07   AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.440E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   9.400E-09
          TIME INCREMENT MAY NOW INCREASE TO     1.08    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.718    ,  FRACTION OF STEP COMPLETED  1.077E-02
 STEP TIME COMPLETED        2.15    ,  TOTAL TIME COMPLETED         2.29    


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   1.08    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.546       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.546    
 LARGEST INCREMENT OF DISP.         1.482E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -4.147E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.228E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -3.754E-02   AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.754E-02
 LARGEST INCREMENT OF DISP.         1.483E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        8.444E-07   AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.719E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.293E-08
          TIME INCREMENT MAY NOW INCREASE TO     1.62    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   1.08    ,  FRACTION OF STEP COMPLETED  1.616E-02
 STEP TIME COMPLETED        3.23    ,  TOTAL TIME COMPLETED         3.37    


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   1.62    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.586       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.586    
 LARGEST INCREMENT OF DISP.         1.738E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -4.861E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.229E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -7.020E-02   AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -7.020E-02
 LARGEST INCREMENT OF DISP.         1.739E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        9.871E-07   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.777E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.778E-08
          TIME INCREMENT MAY NOW INCREASE TO     2.42    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   1.62    ,  FRACTION OF STEP COMPLETED  2.424E-02
 STEP TIME COMPLETED        4.85    ,  TOTAL TIME COMPLETED         4.99    


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   2.42    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.635       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.635    
 LARGEST INCREMENT OF DISP.         2.038E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.705E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.229E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -8.594E-02   AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.594E-02
 LARGEST INCREMENT OF DISP.         2.039E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.383E-06   AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.863E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.449E-08
          TIME INCREMENT MAY NOW INCREASE TO     3.64    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   2.42    ,  FRACTION OF STEP COMPLETED  3.636E-02
 STEP TIME COMPLETED        7.27    ,  TOTAL TIME COMPLETED         7.41    


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   3.64    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.691       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.691    
 LARGEST INCREMENT OF DISP.         2.391E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -6.678E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.229E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      1.698E-02   AT NODE        372   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.698E-02
 LARGEST INCREMENT OF DISP.         2.392E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.261E-06   AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.220E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.366E-08
          TIME INCREMENT MAY NOW INCREASE TO     5.45    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   3.64    ,  FRACTION OF STEP COMPLETED  5.454E-02
 STEP TIME COMPLETED        10.9    ,  TOTAL TIME COMPLETED         11.0    


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   5.45    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       296.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.756       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.756    
 LARGEST INCREMENT OF DISP.         2.803E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -7.845E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.230E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       296.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -8.709E-02   AT NODE        344   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.709E-02
 LARGEST INCREMENT OF DISP.         2.805E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.927E-06   AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.646E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.593E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.642E-08
          TIME INCREMENT MAY NOW INCREASE TO     8.18    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   5.45    ,  FRACTION OF STEP COMPLETED  8.181E-02
 STEP TIME COMPLETED        16.4    ,  TOTAL TIME COMPLETED         16.5    


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   8.18    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.832       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.832    
 LARGEST INCREMENT OF DISP.         3.289E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -9.189E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.647E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.230E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.146       AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.146    
 LARGEST INCREMENT OF DISP.         3.291E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.247E-06   AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.647E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.111E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   6.386E-08
          TIME INCREMENT MAY NOW INCREASE TO     12.3    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   8.18    ,  FRACTION OF STEP COMPLETED  0.123    
 STEP TIME COMPLETED        24.5    ,  TOTAL TIME COMPLETED         24.7    


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   12.3    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      0.922       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.922    
 LARGEST INCREMENT OF DISP.         3.858E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.078E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.647E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.231E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE     -0.168       AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.168    
 LARGEST INCREMENT OF DISP.         3.860E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.405E-06   AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.647E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.638E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   8.815E-08
          TIME INCREMENT MAY NOW INCREASE TO     18.4    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   12.3    ,  FRACTION OF STEP COMPLETED  0.184    
 STEP TIME COMPLETED        36.8    ,  TOTAL TIME COMPLETED         37.0    


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   18.4    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE       1.01       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       1.01    
 LARGEST INCREMENT OF DISP.         4.526E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.264E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.647E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.232E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        294.    
 LARGEST SCALED RESIDUAL FORCE      6.817E-02   AT NODE        359   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      6.817E-02
 LARGEST INCREMENT OF DISP.         4.529E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.316E-06   AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.647E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.537E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.211E-07
          TIME INCREMENT MAY NOW INCREASE TO     27.6    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   18.4    ,  FRACTION OF STEP COMPLETED  0.276    
 STEP TIME COMPLETED        55.2    ,  TOTAL TIME COMPLETED         55.4    


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   27.6    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        295.    
 LARGEST SCALED RESIDUAL FORCE       1.15       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       1.15    
 LARGEST INCREMENT OF DISP.         5.296E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.499E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.647E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.232E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        295.    
 LARGEST SCALED RESIDUAL FORCE     -0.112       AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.112    
 LARGEST INCREMENT OF DISP.         5.315E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.959E-05   AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.647E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.535E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.670E-07
          TIME INCREMENT MAY NOW INCREASE TO     41.4    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   27.6    ,  FRACTION OF STEP COMPLETED  0.414    
 STEP TIME COMPLETED        82.8    ,  TOTAL TIME COMPLETED         83.0    


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   41.4    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  10ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        295.    
 LARGEST SCALED RESIDUAL FORCE       1.28       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       1.28    
 LARGEST INCREMENT OF DISP.         6.233E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.740E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.648E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.233E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        295.    
 LARGEST SCALED RESIDUAL FORCE     -0.231       AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.231    
 LARGEST INCREMENT OF DISP.         6.238E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        6.016E-06   AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.648E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.352E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.304E-07
          TIME INCREMENT MAY NOW INCREASE TO     62.1    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   41.4    ,  FRACTION OF STEP COMPLETED  0.621    
 STEP TIME COMPLETED        124.    ,  TOTAL TIME COMPLETED         124.    


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   62.1    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        295.    
 LARGEST SCALED RESIDUAL FORCE       1.46       AT NODE        360   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       1.46    
 LARGEST INCREMENT OF DISP.         7.336E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.022E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.648E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.234E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       295.       TIME AVG. FORCE        295.    
 LARGEST SCALED RESIDUAL FORCE     -0.294       AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.294    
 LARGEST INCREMENT OF DISP.         7.339E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.873E-06   AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.648E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.647E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.190E-07

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   62.1    ,  FRACTION OF STEP COMPLETED  0.932    
 STEP TIME COMPLETED        186.    ,  TOTAL TIME COMPLETED         187.    


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   13.6    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       295.       TIME AVG. FORCE        295.    
 LARGEST SCALED RESIDUAL FORCE     -2.328E-04   AT NODE       1340   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.328E-04
 LARGEST INCREMENT OF DISP.         1.401E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.082E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 ESTIMATE OF DISP.    CORRECTION   -1.576E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       137.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.648E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.287E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.281E-08

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   13.6    ,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED        200.    ,  TOTAL TIME COMPLETED         200.    


          THE ANALYSIS HAS BEEN COMPLETED



     ANALYSIS SUMMARY:
     TOTAL OF         49  INCREMENTS
                       0  CUTBACKS IN AUTOMATIC INCREMENTATION
                     120  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     120  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                     120  INVOLVE MATRIX DECOMPOSITION, INCLUDING
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
       USER TIME (SEC)      =      2.9    
       SYSTEM TIME (SEC)    =     0.40    
       TOTAL CPU TIME (SEC) =      3.3    
       WALLCLOCK TIME (SEC) =           10
       MEMORY PEAK (GB)     =            0
