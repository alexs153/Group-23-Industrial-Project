1

   Abaqus 2024                                  Date 20-Mar-2025   Time 15:25:30
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
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       150.       TIME AVG. FORCE        150.    
 LARGEST SCALED RESIDUAL FORCE      -489.       AT NODE         96   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -489.    
 LARGEST INCREMENT OF DISP.         7.103E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        7.103E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       69.9    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.407E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.407E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       139.       TIME AVG. FORCE        139.    
 LARGEST SCALED RESIDUAL FORCE      -736.       AT NODE        347   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -736.    
 LARGEST INCREMENT OF DISP.         0.656       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        0.586       AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       64.6    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.776E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 627 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       181.       TIME AVG. FORCE        181.    
 LARGEST SCALED RESIDUAL FORCE      1.024E+03   AT NODE        254   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.024E+03
 LARGEST INCREMENT OF DISP.          1.89       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.         1.23       AT NODE        359   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       84.3    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.323E-36   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 640 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       162.       TIME AVG. FORCE        162.    
 LARGEST SCALED RESIDUAL FORCE      -57.9       AT NODE       1513   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -57.9    
 LARGEST INCREMENT OF DISP.          1.93       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        4.683E-02   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       75.2    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.695E-37   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 642 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       162.       TIME AVG. FORCE        162.    
 LARGEST SCALED RESIDUAL FORCE      0.473       AT NODE        640   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.473    
 LARGEST INCREMENT OF DISP.          1.93       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.226E-03   AT NODE        260   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       75.2    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.158E-37   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.200    ,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED       0.200    ,  TOTAL TIME COMPLETED        0.200    
1

   Abaqus 2024                                  Date 20-Mar-2025   Time 15:25:30
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
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.158       AT NODE        101   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.158    
 LARGEST INCREMENT OF DISP.         2.433E-07   AT NODE        337   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.433E-07   AT NODE        337   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -6.062E-02   AT NODE       1498   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -6.062E-02
 LARGEST INCREMENT OF DISP.         1.937E-06   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.935E-06   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.273E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -3.154E-02   AT NODE        195   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.154E-02
 LARGEST INCREMENT OF DISP.         6.225E-06   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        4.326E-06   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.366E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -3.526E-02   AT NODE        892   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.526E-02
 LARGEST INCREMENT OF DISP.         9.394E-06   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.196E-06   AT NODE        109   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.729E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -4.555E-03   AT NODE       1020   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.555E-03
 LARGEST INCREMENT OF DISP.         1.151E-05   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.147E-06   AT NODE        108   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.379E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -2.180E-03   AT NODE         97   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.180E-03
 LARGEST INCREMENT OF DISP.         1.169E-05   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.805E-07   AT NODE        106   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    8.640E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.067E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.740E-15

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
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -3.177E-03   AT NODE        352   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.177E-03
 LARGEST INCREMENT OF DISP.         7.469E-08   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.162E-05   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.047E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -8.534E-04   AT NODE        960   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.534E-04
 LARGEST INCREMENT OF DISP.         8.041E-08   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.453E-08   AT NODE        328   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.183E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      6.507E-04   AT NODE       1400   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      6.507E-04
 LARGEST INCREMENT OF DISP.         7.187E-08   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -8.716E-09   AT NODE        110   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.551E-43   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      6.962E-04   AT NODE       1408   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      6.962E-04
 LARGEST INCREMENT OF DISP.         6.554E-08   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -6.469E-09   AT NODE        108   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.325E-43   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -7.943E-04   AT NODE       1320   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -7.943E-04
 LARGEST INCREMENT OF DISP.         6.082E-08   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -4.782E-09   AT NODE        109   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.825E-44   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     2 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  2.500E-07
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      2.011E-03   AT NODE        608   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.011E-03
 LARGEST INCREMENT OF DISP.         3.058E-08   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.892E-06   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.058E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      5.243E-04   AT NODE       1335   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.243E-04
 LARGEST INCREMENT OF DISP.         4.337E-08   AT NODE         98   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        2.180E-08   AT NODE        281   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.140E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      7.383E-04   AT NODE        247   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      7.383E-04
 LARGEST INCREMENT OF DISP.         4.079E-08   AT NODE         98   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.685E-09   AT NODE        589   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.898E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      7.228E-04   AT NODE       1287   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      7.228E-04
 LARGEST INCREMENT OF DISP.         3.906E-08   AT NODE        281   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.258E-09   AT NODE        279   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.799E-44   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      8.264E-04   AT NODE       1280   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      8.264E-04
 LARGEST INCREMENT OF DISP.         3.835E-08   AT NODE        281   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.094E-09   AT NODE       1054   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.341E-43   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     2 STARTS. ATTEMPT NUMBER  3, TIME INCREMENT  6.250E-08
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -1.259E-03   AT NODE         97   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.259E-03
 LARGEST INCREMENT OF DISP.         3.655E-09   AT NODE        281   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.655E-09   AT NODE        281   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND SMALL MAGNITUDE
           OF THE INCREMENT OF DISPLACEMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.659E-17

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.250E-08,  FRACTION OF STEP COMPLETED  5.312E-09
 STEP TIME COMPLETED       1.062E-06,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.250E-08
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -8.870E-04   AT NODE       1047   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.870E-04
 LARGEST INCREMENT OF DISP.         5.603E-09   AT NODE        281   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.338E-09   AT NODE        319   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.110E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      1.035E-06   AT NODE       1038   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.035E-06
 LARGEST INCREMENT OF DISP.         8.180E-09   AT NODE        281   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.147E-09   AT NODE       1055   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.547E-43   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.935E-17
          TIME INCREMENT MAY NOW INCREASE TO    9.375E-08

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.250E-08,  FRACTION OF STEP COMPLETED  5.625E-09
 STEP TIME COMPLETED       1.125E-06,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  9.375E-08
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -5.648E-04   AT NODE       1055   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -5.648E-04
 LARGEST INCREMENT OF DISP.         4.253E-09   AT NODE        623   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -8.948E-09   AT NODE        281   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND SMALL MAGNITUDE
           OF THE INCREMENT OF DISPLACEMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.587E-17
          TIME INCREMENT MAY NOW INCREASE TO    1.875E-07

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  9.375E-08,  FRACTION OF STEP COMPLETED  6.094E-09
 STEP TIME COMPLETED       1.219E-06,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.875E-07
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      5.698E-04   AT NODE       1047   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.698E-04
 LARGEST INCREMENT OF DISP.         2.056E-09   AT NODE        637   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -6.574E-09   AT NODE        623   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND SMALL MAGNITUDE
           OF THE INCREMENT OF DISPLACEMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.403E-17
          TIME INCREMENT MAY NOW INCREASE TO    3.750E-07

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.875E-07,  FRACTION OF STEP COMPLETED  7.031E-09
 STEP TIME COMPLETED       1.406E-06,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.750E-07
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      4.016E-04   AT NODE        590   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      4.016E-04
 LARGEST INCREMENT OF DISP.         5.674E-10   AT NODE       1020   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -3.737E-09   AT NODE        637   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND SMALL MAGNITUDE
           OF THE INCREMENT OF DISPLACEMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   6.251E-17
          TIME INCREMENT MAY NOW INCREASE TO    7.500E-07

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.750E-07,  FRACTION OF STEP COMPLETED  8.906E-09
 STEP TIME COMPLETED       1.781E-06,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.500E-07
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -7.684E-07   AT NODE       1055   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -7.684E-07
 LARGEST INCREMENT OF DISP.         1.331E-09   AT NODE         92   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.143E-09   AT NODE        316   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.073E-16
          TIME INCREMENT MAY NOW INCREASE TO    1.500E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.500E-07,  FRACTION OF STEP COMPLETED  1.266E-08
 STEP TIME COMPLETED       2.531E-06,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.500E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -7.619E-07   AT NODE       1037   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -7.619E-07
 LARGEST INCREMENT OF DISP.         3.692E-09   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.873E-09   AT NODE        620   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.045E-18
          TIME INCREMENT MAY NOW INCREASE TO    3.000E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.500E-06,  FRACTION OF STEP COMPLETED  2.016E-08
 STEP TIME COMPLETED       4.031E-06,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      3.640E-09   AT NODE       1037   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      3.640E-09
 LARGEST INCREMENT OF DISP.         7.385E-09   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.406E-12   AT NODE        319   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.200E-18
          TIME INCREMENT MAY NOW INCREASE TO    6.000E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.000E-06,  FRACTION OF STEP COMPLETED  3.516E-08
 STEP TIME COMPLETED       7.031E-06,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -1.644E-09   AT NODE       1037   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.644E-09
 LARGEST INCREMENT OF DISP.         1.477E-08   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.082E-12   AT NODE        319   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.678E-17
          TIME INCREMENT MAY NOW INCREASE TO    1.200E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.000E-06,  FRACTION OF STEP COMPLETED  6.516E-08
 STEP TIME COMPLETED       1.303E-05,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.200E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -2.535E-11   AT NODE        432   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.535E-11
 LARGEST INCREMENT OF DISP.         2.954E-08   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -8.079E-13   AT NODE        577   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   6.712E-17
          TIME INCREMENT MAY NOW INCREASE TO    2.400E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.200E-05,  FRACTION OF STEP COMPLETED  1.252E-07
 STEP TIME COMPLETED       2.503E-05,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.400E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  10ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      2.751E-11   AT NODE        480   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.751E-11
 LARGEST INCREMENT OF DISP.         5.907E-08   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.231E-12   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.685E-16
          TIME INCREMENT MAY NOW INCREASE TO    4.800E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.400E-05,  FRACTION OF STEP COMPLETED  2.452E-07
 STEP TIME COMPLETED       4.903E-05,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.800E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      2.444E-11   AT NODE        470   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.444E-11
 LARGEST INCREMENT OF DISP.         1.181E-07   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.292E-11   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.074E-15
          TIME INCREMENT MAY NOW INCREASE TO    9.600E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.800E-05,  FRACTION OF STEP COMPLETED  4.852E-07
 STEP TIME COMPLETED       9.703E-05,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  9.600E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -2.728E-11   AT NODE          1   DOF  2
   INSTANCE:                                                                                 
  CORRESPONDING RESIDUAL FORCE     -2.728E-11
 LARGEST INCREMENT OF DISP.         2.362E-07   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.166E-11   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.293E-15
          TIME INCREMENT MAY NOW INCREASE TO    1.920E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  9.600E-05,  FRACTION OF STEP COMPLETED  9.652E-07
 STEP TIME COMPLETED       1.930E-04,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.920E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.336       AT NODE        263   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.336    
 LARGEST INCREMENT OF DISP.         7.584E-05   AT NODE          5   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        7.538E-05   AT NODE          5   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      0.161       AT NODE        191   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.161    
 LARGEST INCREMENT OF DISP.         7.877E-05   AT NODE        110   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.323E-06   AT NODE         85   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.384E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      5.260E-02   AT NODE       1461   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.260E-02
 LARGEST INCREMENT OF DISP.         8.284E-05   AT NODE        110   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        4.291E-06   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.091E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -1.707E-02   AT NODE        603   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.707E-02
 LARGEST INCREMENT OF DISP.         8.657E-05   AT NODE        108   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.802E-06   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.726E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -4.420E-03   AT NODE       1484   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.420E-03
 LARGEST INCREMENT OF DISP.         8.694E-05   AT NODE        108   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.980E-07   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.871E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.947E-10

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.920E-04,  FRACTION OF STEP COMPLETED  1.925E-06
 STEP TIME COMPLETED       3.850E-04,  TOTAL TIME COMPLETED        0.200    


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.920E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      2.370E-02   AT NODE        597   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.370E-02
 LARGEST INCREMENT OF DISP.         3.677E-05   AT NODE        104   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -5.107E-05   AT NODE       1441   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.080E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -2.176E-02   AT NODE        592   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.176E-02
 LARGEST INCREMENT OF DISP.         3.670E-05   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -5.094E-07   AT NODE        264   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.125E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -3.820E-03   AT NODE         96   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.820E-03
 LARGEST INCREMENT OF DISP.         3.679E-05   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        8.528E-08   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -7.269E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   7.579E-10

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.920E-04,  FRACTION OF STEP COMPLETED  2.885E-06
 STEP TIME COMPLETED       5.770E-04,  TOTAL TIME COMPLETED        0.201    


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.920E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.829       AT NODE         98   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.829    
 LARGEST INCREMENT OF DISP.         1.388E-05   AT NODE       1409   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.486E-05   AT NODE        105   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.371E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -8.409E-02   AT NODE        607   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.409E-02
 LARGEST INCREMENT OF DISP.         2.042E-05   AT NODE         99   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        8.443E-06   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.686E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      4.535E-02   AT NODE       1482   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      4.535E-02
 LARGEST INCREMENT OF DISP.         2.171E-05   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.328E-06   AT NODE        637   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.126E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -3.663E-03   AT NODE        352   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.663E-03
 LARGEST INCREMENT OF DISP.         2.224E-05   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        5.392E-07   AT NODE        100   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION   -4.356E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.593E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.457E-10
          TIME INCREMENT MAY NOW INCREASE TO    2.880E-04

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.920E-04,  FRACTION OF STEP COMPLETED  3.845E-06
 STEP TIME COMPLETED       7.690E-04,  TOTAL TIME COMPLETED        0.201    


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.880E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.792       AT NODE         97   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.792    
 LARGEST INCREMENT OF DISP.         1.854E-05   AT NODE        266   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.680E-05   AT NODE         85   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -6.647E-02   AT NODE         99   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -6.647E-02
 LARGEST INCREMENT OF DISP.         2.367E-05   AT NODE        622   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        7.096E-06   AT NODE         85   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -7.559E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -1.674E-02   AT NODE       1483   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.674E-02
 LARGEST INCREMENT OF DISP.         2.445E-05   AT NODE         85   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        8.078E-07   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    2.035E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.318E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.540E-10
          TIME INCREMENT MAY NOW INCREASE TO    4.320E-04

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.880E-04,  FRACTION OF STEP COMPLETED  5.285E-06
 STEP TIME COMPLETED       1.057E-03,  TOTAL TIME COMPLETED        0.201    


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.320E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.504       AT NODE        349   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.504    
 LARGEST INCREMENT OF DISP.         2.390E-05   AT NODE        631   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.292E-05   AT NODE        616   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -3.073E-02   AT NODE        351   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.073E-02
 LARGEST INCREMENT OF DISP.         2.792E-05   AT NODE        631   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        4.090E-06   AT NODE         85   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.196E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -8.060E-03   AT NODE        598   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.060E-03
 LARGEST INCREMENT OF DISP.         2.881E-05   AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        9.098E-07   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    2.386E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.526E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.461E-10
          TIME INCREMENT MAY NOW INCREASE TO    6.480E-04

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.320E-04,  FRACTION OF STEP COMPLETED  7.445E-06
 STEP TIME COMPLETED       1.489E-03,  TOTAL TIME COMPLETED        0.201    


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.480E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.641       AT NODE         99   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.641    
 LARGEST INCREMENT OF DISP.         2.786E-05   AT NODE          5   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.590E-05   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -2.269E-02   AT NODE        351   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.269E-02
 LARGEST INCREMENT OF DISP.         3.245E-05   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        5.129E-06   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.034E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -1.246E-02   AT NODE       1481   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.246E-02
 LARGEST INCREMENT OF DISP.         3.343E-05   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.016E-06   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.405E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      2.855E-03   AT NODE         93   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.855E-03
 LARGEST INCREMENT OF DISP.         3.388E-05   AT NODE         80   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        4.740E-07   AT NODE        110   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION   -1.086E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.890E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.762E-10
          TIME INCREMENT MAY NOW INCREASE TO    9.720E-04

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.480E-04,  FRACTION OF STEP COMPLETED  1.069E-05
 STEP TIME COMPLETED       2.137E-03,  TOTAL TIME COMPLETED        0.202    


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  9.720E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.622       AT NODE        100   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.622    
 LARGEST INCREMENT OF DISP.         3.255E-05   AT NODE          5   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.889E-05   AT NODE        632   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -2.819E-02   AT NODE         96   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.819E-02
 LARGEST INCREMENT OF DISP.         3.794E-05   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        5.932E-06   AT NODE        632   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.122E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -8.853E-03   AT NODE       1452   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.853E-03
 LARGEST INCREMENT OF DISP.         3.852E-05   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        5.997E-07   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    1.883E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.315E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.508E-10
          TIME INCREMENT MAY NOW INCREASE TO    1.458E-03

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  9.720E-04,  FRACTION OF STEP COMPLETED  1.555E-05
 STEP TIME COMPLETED       3.109E-03,  TOTAL TIME COMPLETED        0.203    


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.458E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.558       AT NODE        101   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.558    
 LARGEST INCREMENT OF DISP.         4.108E-05   AT NODE        110   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.705E-05   AT NODE        636   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -4.900E-02   AT NODE         98   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.900E-02
 LARGEST INCREMENT OF DISP.         4.482E-05   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.875E-06   AT NODE        637   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.268E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -1.307E-02   AT NODE        589   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.307E-02
 LARGEST INCREMENT OF DISP.         4.593E-05   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.130E-06   AT NODE         99   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    3.013E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.690E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.054E-10
          TIME INCREMENT MAY NOW INCREASE TO    2.187E-03

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.458E-03,  FRACTION OF STEP COMPLETED  2.284E-05
 STEP TIME COMPLETED       4.567E-03,  TOTAL TIME COMPLETED        0.205    


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.187E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.589       AT NODE        100   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.589    
 LARGEST INCREMENT OF DISP.         4.935E-05   AT NODE        110   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.060E-05   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -4.129E-02   AT NODE         98   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.129E-02
 LARGEST INCREMENT OF DISP.         5.238E-05   AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        3.875E-06   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    3.706E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.019E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.574E-10
          TIME INCREMENT MAY NOW INCREASE TO    3.280E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.187E-03,  FRACTION OF STEP COMPLETED  3.377E-05
 STEP TIME COMPLETED       6.754E-03,  TOTAL TIME COMPLETED        0.207    


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.280E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      0.367       AT NODE        359   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.367    
 LARGEST INCREMENT OF DISP.         6.115E-05   AT NODE        104   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -1.754E-05   AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -4.094E-02   AT NODE         96   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.094E-02
 LARGEST INCREMENT OF DISP.         6.250E-05   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.470E-06   AT NODE        366   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION   -1.640E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.444E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.022E-10
          TIME INCREMENT MAY NOW INCREASE TO    4.921E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.280E-03,  FRACTION OF STEP COMPLETED  5.017E-05
 STEP TIME COMPLETED       1.003E-02,  TOTAL TIME COMPLETED        0.210    


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.921E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.269       AT NODE        100   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.269    
 LARGEST INCREMENT OF DISP.         7.278E-05   AT NODE         61   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.102E-05   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -3.233E-02   AT NODE        887   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.233E-02
 LARGEST INCREMENT OF DISP.         7.441E-05   AT NODE         82   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.650E-06   AT NODE        640   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    1.986E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.076E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   6.546E-10
          TIME INCREMENT MAY NOW INCREASE TO    7.381E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.921E-03,  FRACTION OF STEP COMPLETED  7.478E-05
 STEP TIME COMPLETED       1.496E-02,  TOTAL TIME COMPLETED        0.215    


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.381E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.317       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.317    
 LARGEST INCREMENT OF DISP.         8.527E-05   AT NODE         61   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.642E-05   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -6.953E-02   AT NODE        345   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -6.953E-02
 LARGEST INCREMENT OF DISP.         8.652E-05   AT NODE         59   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.290E-06   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION   -3.314E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.357E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   8.166E-10
          TIME INCREMENT MAY NOW INCREASE TO    1.107E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.381E-03,  FRACTION OF STEP COMPLETED  1.117E-04
 STEP TIME COMPLETED       2.234E-02,  TOTAL TIME COMPLETED        0.222    


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.107E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.122       AT NODE        100   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.122    
 LARGEST INCREMENT OF DISP.         1.013E-04   AT NODE         63   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -2.871E-05   AT NODE         29   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -1.011E-02   AT NODE        892   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.011E-02
 LARGEST INCREMENT OF DISP.         1.027E-04   AT NODE         63   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.372E-06   AT NODE        356   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    1.135E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.096E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.303E-09
          TIME INCREMENT MAY NOW INCREASE TO    1.661E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.107E-02,  FRACTION OF STEP COMPLETED  1.670E-04
 STEP TIME COMPLETED       3.341E-02,  TOTAL TIME COMPLETED        0.233    


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.661E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.305       AT NODE        347   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.305    
 LARGEST INCREMENT OF DISP.         1.185E-04   AT NODE        574   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -3.580E-05   AT NODE         36   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      2.097E-02   AT NODE       1476   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.097E-02
 LARGEST INCREMENT OF DISP.         1.195E-04   AT NODE        574   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.071E-06   AT NODE         17   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.124E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.846E-10
          TIME INCREMENT MAY NOW INCREASE TO    2.491E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.661E-02,  FRACTION OF STEP COMPLETED  2.501E-04
 STEP TIME COMPLETED       5.002E-02,  TOTAL TIME COMPLETED        0.250    


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.491E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.238       AT NODE        347   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.238    
 LARGEST INCREMENT OF DISP.         1.394E-04   AT NODE        572   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -3.989E-05   AT NODE        574   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -1.420E-02   AT NODE        611   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.420E-02
 LARGEST INCREMENT OF DISP.         1.403E-04   AT NODE        572   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        9.586E-07   AT NODE        639   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -9.481E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.266E-09
          TIME INCREMENT MAY NOW INCREASE TO    3.737E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.491E-02,  FRACTION OF STEP COMPLETED  3.746E-04
 STEP TIME COMPLETED       7.493E-02,  TOTAL TIME COMPLETED        0.275    


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.737E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.216       AT NODE        632   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.216    
 LARGEST INCREMENT OF DISP.         1.639E-04   AT NODE        581   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -4.660E-05   AT NODE         62   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -1.356E-02   AT NODE        620   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.356E-02
 LARGEST INCREMENT OF DISP.         1.646E-04   AT NODE        581   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        7.720E-07   AT NODE        639   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.551E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.288E-09
          TIME INCREMENT MAY NOW INCREASE TO    5.605E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.737E-02,  FRACTION OF STEP COMPLETED  5.615E-04
 STEP TIME COMPLETED       0.112    ,  TOTAL TIME COMPLETED        0.312    


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.605E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.586       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.586    
 LARGEST INCREMENT OF DISP.         1.913E-04   AT NODE        582   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -5.560E-05   AT NODE        574   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -3.649E-02   AT NODE        345   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.649E-02
 LARGEST INCREMENT OF DISP.         1.923E-04   AT NODE        582   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.005E-06   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.608E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.176E-09
          TIME INCREMENT MAY NOW INCREASE TO    8.408E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.605E-02,  FRACTION OF STEP COMPLETED  8.417E-04
 STEP TIME COMPLETED       0.168    ,  TOTAL TIME COMPLETED        0.368    


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.408E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.529       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.529    
 LARGEST INCREMENT OF DISP.         2.260E-04   AT NODE        577   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -6.244E-05   AT NODE        582   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      3.629E-02   AT NODE       1484   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      3.629E-02
 LARGEST INCREMENT OF DISP.         2.265E-04   AT NODE        577   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        5.336E-07   AT NODE        639   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.358E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.530E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.126    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.408E-02,  FRACTION OF STEP COMPLETED  1.262E-03
 STEP TIME COMPLETED       0.252    ,  TOTAL TIME COMPLETED        0.452    


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.126    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.617       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.617    
 LARGEST INCREMENT OF DISP.         2.642E-04   AT NODE        578   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -7.563E-05   AT NODE         60   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -3.351E-02   AT NODE        622   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.351E-02
 LARGEST INCREMENT OF DISP.         2.652E-04   AT NODE         59   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.        1.013E-06   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -9.266E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.633E-10
          TIME INCREMENT MAY NOW INCREASE TO    0.189    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.126    ,  FRACTION OF STEP COMPLETED  1.893E-03
 STEP TIME COMPLETED       0.379    ,  TOTAL TIME COMPLETED        0.579    


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.189    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.688       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.688    
 LARGEST INCREMENT OF DISP.         3.099E-04   AT NODE        577   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 LARGEST CORRECTION TO DISP.       -8.785E-05   AT NODE         59   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      3.854E-02   AT NODE       1484   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      3.854E-02
 LARGEST INCREMENT OF DISP.         3.110E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.152E-06   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.896E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   6.386E-10
          TIME INCREMENT MAY NOW INCREASE TO    0.284    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.189    ,  FRACTION OF STEP COMPLETED  2.839E-03
 STEP TIME COMPLETED       0.568    ,  TOTAL TIME COMPLETED        0.768    


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.284    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.673       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.673    
 LARGEST INCREMENT OF DISP.         3.637E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.028E-04   AT NODE        579   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -4.020E-02   AT NODE        623   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.020E-02
 LARGEST INCREMENT OF DISP.         3.647E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.015E-06   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.116E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   8.737E-10
          TIME INCREMENT MAY NOW INCREASE TO    0.426    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.284    ,  FRACTION OF STEP COMPLETED  4.257E-03
 STEP TIME COMPLETED       0.851    ,  TOTAL TIME COMPLETED         1.05    


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.426    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.759       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.759    
 LARGEST INCREMENT OF DISP.         4.264E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.207E-04   AT NODE         59   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      3.664E-02   AT NODE       1485   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      3.664E-02
 LARGEST INCREMENT OF DISP.         4.276E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.277E-06   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.460E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.201E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.638    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.426    ,  FRACTION OF STEP COMPLETED  6.385E-03
 STEP TIME COMPLETED        1.28    ,  TOTAL TIME COMPLETED         1.48    


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.638    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.803       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.803    
 LARGEST INCREMENT OF DISP.         5.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.414E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      4.158E-02   AT NODE       1486   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      4.158E-02
 LARGEST INCREMENT OF DISP.         5.014E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.492E-06   AT NODE        648   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.479E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.646E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.958    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.638    ,  FRACTION OF STEP COMPLETED  9.578E-03
 STEP TIME COMPLETED        1.92    ,  TOTAL TIME COMPLETED         2.12    


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.958    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.855       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.855    
 LARGEST INCREMENT OF DISP.         5.861E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.660E-04   AT NODE         59   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      5.035E-02   AT NODE       1485   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.035E-02
 LARGEST INCREMENT OF DISP.         5.875E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.395E-06   AT NODE        102   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.225E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.269E-09
          TIME INCREMENT MAY NOW INCREASE TO     1.44    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.958    ,  FRACTION OF STEP COMPLETED  1.437E-02
 STEP TIME COMPLETED        2.87    ,  TOTAL TIME COMPLETED         3.07    


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   1.44    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.914       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.914    
 LARGEST INCREMENT OF DISP.         6.876E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.936E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.055E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -1.732E-02   AT NODE        634   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.732E-02
 LARGEST INCREMENT OF DISP.         6.893E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.749E-06   AT NODE        648   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.612E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.112E-09
          TIME INCREMENT MAY NOW INCREASE TO     2.15    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   1.44    ,  FRACTION OF STEP COMPLETED  2.155E-02
 STEP TIME COMPLETED        4.31    ,  TOTAL TIME COMPLETED         4.51    


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   2.15    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.986       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.986    
 LARGEST INCREMENT OF DISP.         8.060E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.280E-04   AT NODE         59   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.056E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -3.744E-02   AT NODE       1489   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.744E-02
 LARGEST INCREMENT OF DISP.         8.076E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.657E-06   AT NODE        648   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.398E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.289E-09
          TIME INCREMENT MAY NOW INCREASE TO     3.23    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   2.15    ,  FRACTION OF STEP COMPLETED  3.232E-02
 STEP TIME COMPLETED        6.46    ,  TOTAL TIME COMPLETED         6.66    


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   3.23    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      -1.07       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.07    
 LARGEST INCREMENT OF DISP.         9.455E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.660E-04   AT NODE         59   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.056E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -4.713E-02   AT NODE       1493   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.713E-02
 LARGEST INCREMENT OF DISP.         9.474E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.878E-06   AT NODE        648   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.631E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.896E-09
          TIME INCREMENT MAY NOW INCREASE TO     4.85    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   3.23    ,  FRACTION OF STEP COMPLETED  4.848E-02
 STEP TIME COMPLETED        9.70    ,  TOTAL TIME COMPLETED         9.90    


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   4.85    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      -1.16       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.16    
 LARGEST INCREMENT OF DISP.         1.109E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.124E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.056E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -5.583E-02   AT NODE       1490   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -5.583E-02
 LARGEST INCREMENT OF DISP.         1.111E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.106E-06   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.187E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   8.102E-09
          TIME INCREMENT MAY NOW INCREASE TO     7.27    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   4.85    ,  FRACTION OF STEP COMPLETED  7.272E-02
 STEP TIME COMPLETED        14.5    ,  TOTAL TIME COMPLETED         14.7    


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.27    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      -1.27       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.27    
 LARGEST INCREMENT OF DISP.         1.300E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.657E-04   AT NODE         59   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.056E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -6.834E-02   AT NODE       1490   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -6.834E-02
 LARGEST INCREMENT OF DISP.         1.302E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.152E-06   AT NODE        648   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.411E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.114E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.117E-08
          TIME INCREMENT MAY NOW INCREASE TO     10.9    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.27    ,  FRACTION OF STEP COMPLETED  0.109    
 STEP TIME COMPLETED        21.8    ,  TOTAL TIME COMPLETED         22.0    


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   10.9    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      -1.50       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.50    
 LARGEST INCREMENT OF DISP.         1.524E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -4.295E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.412E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.056E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -8.113E-02   AT NODE       1490   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.113E-02
 LARGEST INCREMENT OF DISP.         1.527E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.258E-06   AT NODE        640   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.412E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.903E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.537E-08
          TIME INCREMENT MAY NOW INCREASE TO     16.4    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   10.9    ,  FRACTION OF STEP COMPLETED  0.164    
 STEP TIME COMPLETED        32.7    ,  TOTAL TIME COMPLETED         32.9    


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   16.4    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      -1.54       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.54    
 LARGEST INCREMENT OF DISP.         1.788E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.025E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.412E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.057E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -9.576E-02   AT NODE       1490   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -9.576E-02
 LARGEST INCREMENT OF DISP.         1.791E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.055E-06   AT NODE        648   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.412E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.084E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.118E-08
          TIME INCREMENT MAY NOW INCREASE TO     24.5    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   16.4    ,  FRACTION OF STEP COMPLETED  0.245    
 STEP TIME COMPLETED        49.1    ,  TOTAL TIME COMPLETED         49.3    


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   24.5    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      -1.84       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.84    
 LARGEST INCREMENT OF DISP.         2.096E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.909E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.412E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.057E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.111       AT NODE       1490   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.111    
 LARGEST INCREMENT OF DISP.         2.100E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        4.562E-06   AT NODE        640   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.412E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.019E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.915E-08
          TIME INCREMENT MAY NOW INCREASE TO     36.8    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   24.5    ,  FRACTION OF STEP COMPLETED  0.368    
 STEP TIME COMPLETED        73.6    ,  TOTAL TIME COMPLETED         73.8    


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   36.8    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      -1.91       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.91    
 LARGEST INCREMENT OF DISP.         2.461E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -6.901E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.412E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.058E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       247.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.130       AT NODE       1490   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.130    
 LARGEST INCREMENT OF DISP.         2.464E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.522E-06   AT NODE          1   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.412E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.338E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.014E-08
          TIME INCREMENT MAY NOW INCREASE TO     55.2    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   36.8    ,  FRACTION OF STEP COMPLETED  0.552    
 STEP TIME COMPLETED        110.    ,  TOTAL TIME COMPLETED         111.    


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   55.2    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       248.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      -2.31       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -2.31    
 LARGEST INCREMENT OF DISP.         2.885E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -8.111E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.412E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.058E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       248.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.149       AT NODE       1490   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.149    
 LARGEST INCREMENT OF DISP.         2.890E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        5.528E-06   AT NODE        640   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.412E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.388E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.557E-08

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   55.2    ,  FRACTION OF STEP COMPLETED  0.828    
 STEP TIME COMPLETED        166.    ,  TOTAL TIME COMPLETED         166.    


  INCREMENT    49 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   34.3    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       248.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE      -1.68       AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -1.68    
 LARGEST INCREMENT OF DISP.         1.506E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.907E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.412E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.345E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       248.       TIME AVG. FORCE        247.    
 LARGEST SCALED RESIDUAL FORCE     -0.113       AT NODE       1489   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.113    
 LARGEST INCREMENT OF DISP.         1.508E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.020E-06   AT NODE        640   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       115.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.412E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.120E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.607E-08

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   34.3    ,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED        200.    ,  TOTAL TIME COMPLETED         200.    


          THE ANALYSIS HAS BEEN COMPLETED



     ANALYSIS SUMMARY:
     TOTAL OF         50  INCREMENTS
                       2  CUTBACKS IN AUTOMATIC INCREMENTATION
                     117  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     117  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                     117  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       2  WARNING MESSAGES DURING USER INPUT PROCESSING
                       3  WARNING MESSAGES DURING ANALYSIS
                       0  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                       0  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =      1.4    
       SYSTEM TIME (SEC)    =     0.10    
       TOTAL CPU TIME (SEC) =      1.5    
       WALLCLOCK TIME (SEC) =            9
       MEMORY PEAK (GB)     =            0
