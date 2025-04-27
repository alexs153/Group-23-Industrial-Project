1

   Abaqus 2024                                  Date 20-Mar-2025   Time 09:22:09
   For use by UNIVERSITY OF BRISTOL under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     S T A T I C   A N A L Y S I S


                                                                                          

     AUTOMATIC TIME CONTROL WITH -
          A SUGGESTED INITIAL TIME INCREMENT OF                0.160    
          AND A TOTAL TIME PERIOD OF                           0.160    
          THE MINIMUM TIME INCREMENT ALLOWED IS                1.600E-06
          THE MAXIMUM TIME INCREMENT ALLOWED IS                0.160    

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


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.160    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       184.       TIME AVG. FORCE        184.    
 LARGEST SCALED RESIDUAL FORCE      -743.       AT NODE         96   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      -743.    
 LARGEST INCREMENT OF DISP.         8.722E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        8.722E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       85.5    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.759E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.759E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       173.       TIME AVG. FORCE        173.    
 LARGEST SCALED RESIDUAL FORCE     -1.393E+03   AT NODE        346   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.393E+03
 LARGEST INCREMENT OF DISP.          1.17       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.         1.09       AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       80.2    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.764E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.209E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 703 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       239.       TIME AVG. FORCE        239.    
 LARGEST SCALED RESIDUAL FORCE      2.614E+03   AT NODE        200   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.614E+03
 LARGEST INCREMENT OF DISP.          2.97       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.         1.79       AT NODE         81   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       111.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.763E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.765E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 705 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       205.       TIME AVG. FORCE        205.    
 LARGEST SCALED RESIDUAL FORCE       149.       AT NODE        244   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       149.    
 LARGEST INCREMENT OF DISP.          3.39       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        0.429       AT NODE         74   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       95.1    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.490E-36   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 706 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       204.       TIME AVG. FORCE        204.    
 LARGEST SCALED RESIDUAL FORCE       8.71       AT NODE        236   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       8.71    
 LARGEST INCREMENT OF DISP.          3.47       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        7.333E-02   AT NODE        398   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       94.7    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.308E-36   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

 ***WARNING: THE STRAIN INCREMENT HAS EXCEEDED FIFTY TIMES THE STRAIN TO CAUSE 
             FIRST YIELD AT 706 POINTS

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                       204.       TIME AVG. FORCE        204.    
 LARGEST SCALED RESIDUAL FORCE      5.507E-02   AT NODE        258   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.507E-02
 LARGEST INCREMENT OF DISP.          3.47       AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        5.744E-04   AT NODE          6   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       94.7    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.215E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.160    ,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED       0.160    ,  TOTAL TIME COMPLETED        0.160    
1

   Abaqus 2024                                  Date 20-Mar-2025   Time 09:22:10
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


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -4.179E-02   AT NODE        348   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.179E-02
 LARGEST INCREMENT OF DISP.         3.666E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.666E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -0.211       AT NODE          6   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.211    
 LARGEST INCREMENT OF DISP.         8.907E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        5.243E-05   AT NODE         78   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.881E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -2.022E-04   AT NODE          6   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.022E-04
 LARGEST INCREMENT OF DISP.         1.046E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.560E-05   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    7.548E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.721E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.293E-11

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-06,  FRACTION OF STEP COMPLETED  5.000E-09
 STEP TIME COMPLETED       1.000E-06,  TOTAL TIME COMPLETED        0.160    


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -2.857E-03   AT NODE         29   DOF  1
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.857E-03
 LARGEST INCREMENT OF DISP.         1.298E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -9.167E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -7.461E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -2.285E-04   AT NODE        641   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.285E-04
 LARGEST INCREMENT OF DISP.         1.321E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.269E-07   AT NODE        432   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
 ESTIMATE OF DISP.    CORRECTION    1.815E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.595E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.397E-13

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-06,  FRACTION OF STEP COMPLETED  1.000E-08
 STEP TIME COMPLETED       2.000E-06,  TOTAL TIME COMPLETED        0.160    


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      6.130E-04   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      6.130E-04
 LARGEST INCREMENT OF DISP.         9.579E-06   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.632E-06   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.985E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -4.897E-04   AT NODE        373   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.897E-04
 LARGEST INCREMENT OF DISP.         9.580E-06   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        8.521E-10   AT NODE         87   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.255E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.267E-13
          TIME INCREMENT MAY NOW INCREASE TO    1.500E-06

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-06,  FRACTION OF STEP COMPLETED  1.500E-08
 STEP TIME COMPLETED       3.000E-06,  TOTAL TIME COMPLETED        0.160    


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.500E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -2.113E-03   AT NODE        373   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.113E-03
 LARGEST INCREMENT OF DISP.         1.122E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.151E-06   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -8.520E-07   AT NODE        368   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.520E-07
 LARGEST INCREMENT OF DISP.         1.122E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.303E-09   AT NODE         24   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.019E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.126E-13
          TIME INCREMENT MAY NOW INCREASE TO    2.250E-06

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.500E-06,  FRACTION OF STEP COMPLETED  2.250E-08
 STEP TIME COMPLETED       4.500E-06,  TOTAL TIME COMPLETED        0.160    


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.250E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      5.225E-04   AT NODE       1516   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.225E-04
 LARGEST INCREMENT OF DISP.         1.316E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.671E-06   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -2.449E-04   AT NODE        379   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.449E-04
 LARGEST INCREMENT OF DISP.         1.316E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        6.444E-10   AT NODE         86   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.122E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.289E-13
          TIME INCREMENT MAY NOW INCREASE TO    3.375E-06

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.250E-06,  FRACTION OF STEP COMPLETED  3.375E-08
 STEP TIME COMPLETED       6.750E-06,  TOTAL TIME COMPLETED        0.160    


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.375E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -2.518E-03   AT NODE        366   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.518E-03
 LARGEST INCREMENT OF DISP.         1.542E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -4.319E-06   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      7.900E-07   AT NODE        366   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      7.900E-07
 LARGEST INCREMENT OF DISP.         1.542E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.600E-09   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.167E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.902E-13
          TIME INCREMENT MAY NOW INCREASE TO    5.062E-06

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.375E-06,  FRACTION OF STEP COMPLETED  5.062E-08
 STEP TIME COMPLETED       1.012E-05,  TOTAL TIME COMPLETED        0.160    


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.062E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      2.782E-03   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.782E-03
 LARGEST INCREMENT OF DISP.         1.808E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.054E-06   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      7.805E-07   AT NODE         21   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      7.805E-07
 LARGEST INCREMENT OF DISP.         1.808E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        4.501E-09   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.911E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   8.105E-13
          TIME INCREMENT MAY NOW INCREASE TO    7.594E-06

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.062E-06,  FRACTION OF STEP COMPLETED  7.594E-08
 STEP TIME COMPLETED       1.519E-05,  TOTAL TIME COMPLETED        0.160    


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.594E-06
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -3.892E-03   AT NODE         21   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.892E-03
 LARGEST INCREMENT OF DISP.         2.119E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.933E-06   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      1.163E-06   AT NODE         21   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.163E-06
 LARGEST INCREMENT OF DISP.         2.119E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        5.177E-09   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.279E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.115E-12
          TIME INCREMENT MAY NOW INCREASE TO    1.139E-05

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.594E-06,  FRACTION OF STEP COMPLETED  1.139E-07
 STEP TIME COMPLETED       2.278E-05,  TOTAL TIME COMPLETED        0.160    


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.139E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      4.408E-03   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      4.408E-03
 LARGEST INCREMENT OF DISP.         2.484E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -6.949E-06   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      1.220E-06   AT NODE         21   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.220E-06
 LARGEST INCREMENT OF DISP.         2.485E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        7.759E-09   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.532E-42   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.531E-12
          TIME INCREMENT MAY NOW INCREASE TO    1.709E-05

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.139E-05,  FRACTION OF STEP COMPLETED  1.709E-07
 STEP TIME COMPLETED       3.417E-05,  TOTAL TIME COMPLETED        0.160    


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.709E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      5.503E-03   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.503E-03
 LARGEST INCREMENT OF DISP.         2.912E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -8.155E-06   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      6.590E-04   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      6.590E-04
 LARGEST INCREMENT OF DISP.         2.913E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        9.278E-09   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.979E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.105E-12
          TIME INCREMENT MAY NOW INCREASE TO    2.563E-05

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.709E-05,  FRACTION OF STEP COMPLETED  2.563E-07
 STEP TIME COMPLETED       5.126E-05,  TOTAL TIME COMPLETED        0.160    


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.563E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      6.618E-03   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      6.618E-03
 LARGEST INCREMENT OF DISP.         3.414E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -9.554E-06   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      6.152E-04   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      6.152E-04
 LARGEST INCREMENT OF DISP.         3.415E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.168E-08   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.210E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.893E-12
          TIME INCREMENT MAY NOW INCREASE TO    3.844E-05

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.563E-05,  FRACTION OF STEP COMPLETED  3.844E-07
 STEP TIME COMPLETED       7.689E-05,  TOTAL TIME COMPLETED        0.160    


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.844E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      8.176E-03   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      8.176E-03
 LARGEST INCREMENT OF DISP.         4.002E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.121E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      4.377E-04   AT NODE        372   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      4.377E-04
 LARGEST INCREMENT OF DISP.         4.003E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.442E-08   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.163E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.975E-12
          TIME INCREMENT MAY NOW INCREASE TO    5.767E-05

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.844E-05,  FRACTION OF STEP COMPLETED  5.767E-07
 STEP TIME COMPLETED       1.153E-04,  TOTAL TIME COMPLETED        0.160    


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.767E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      9.656E-03   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      9.656E-03
 LARGEST INCREMENT OF DISP.         4.692E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.313E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      8.976E-04   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      8.976E-04
 LARGEST INCREMENT OF DISP.         4.694E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.676E-08   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.769E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.464E-12
          TIME INCREMENT MAY NOW INCREASE TO    8.650E-05

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.767E-05,  FRACTION OF STEP COMPLETED  8.650E-07
 STEP TIME COMPLETED       1.730E-04,  TOTAL TIME COMPLETED        0.160    


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.650E-05
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      1.174E-02   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.174E-02
 LARGEST INCREMENT OF DISP.         5.500E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.540E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -1.034E-03   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.034E-03
 LARGEST INCREMENT OF DISP.         5.502E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.068E-08   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.941E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   7.509E-12
          TIME INCREMENT MAY NOW INCREASE TO    1.297E-04

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.650E-05,  FRACTION OF STEP COMPLETED  1.297E-06
 STEP TIME COMPLETED       2.595E-04,  TOTAL TIME COMPLETED        0.160    


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.297E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      1.389E-02   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.389E-02
 LARGEST INCREMENT OF DISP.         6.448E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.805E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      8.970E-04   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      8.970E-04
 LARGEST INCREMENT OF DISP.         6.451E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.656E-08   AT NODE         84   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.179E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.032E-11
          TIME INCREMENT MAY NOW INCREASE TO    1.946E-04

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.297E-04,  FRACTION OF STEP COMPLETED  1.946E-06
 STEP TIME COMPLETED       3.892E-04,  TOTAL TIME COMPLETED        0.160    


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.946E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      1.671E-02   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.671E-02
 LARGEST INCREMENT OF DISP.         7.559E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.117E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -1.260E-03   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.260E-03
 LARGEST INCREMENT OF DISP.         7.562E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.002E-08   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.879E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.418E-11
          TIME INCREMENT MAY NOW INCREASE TO    2.919E-04

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.946E-04,  FRACTION OF STEP COMPLETED  2.919E-06
 STEP TIME COMPLETED       5.839E-04,  TOTAL TIME COMPLETED        0.161    


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.919E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      1.996E-02   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.996E-02
 LARGEST INCREMENT OF DISP.         8.862E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.481E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      1.158E-03   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.158E-03
 LARGEST INCREMENT OF DISP.         8.865E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.706E-08   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.154E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.949E-11
          TIME INCREMENT MAY NOW INCREASE TO    4.379E-04

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.919E-04,  FRACTION OF STEP COMPLETED  4.379E-06
 STEP TIME COMPLETED       8.758E-04,  TOTAL TIME COMPLETED        0.161    


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.379E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      2.353E-02   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.353E-02
 LARGEST INCREMENT OF DISP.         1.039E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.909E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -1.697E-03   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.697E-03
 LARGEST INCREMENT OF DISP.         1.039E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        4.235E-08   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.919E-41   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.679E-11
          TIME INCREMENT MAY NOW INCREASE TO    6.568E-04

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.379E-04,  FRACTION OF STEP COMPLETED  6.568E-06
 STEP TIME COMPLETED       1.314E-03,  TOTAL TIME COMPLETED        0.161    


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.568E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      2.801E-02   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.801E-02
 LARGEST INCREMENT OF DISP.         1.218E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.410E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      1.570E-03   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.570E-03
 LARGEST INCREMENT OF DISP.         1.218E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        5.167E-08   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.124E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.682E-11
          TIME INCREMENT MAY NOW INCREASE TO    9.853E-04

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.568E-04,  FRACTION OF STEP COMPLETED  9.853E-06
 STEP TIME COMPLETED       1.971E-03,  TOTAL TIME COMPLETED        0.162    


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  9.853E-04
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      3.299E-02   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      3.299E-02
 LARGEST INCREMENT OF DISP.         1.428E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.998E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -2.341E-03   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.341E-03
 LARGEST INCREMENT OF DISP.         1.428E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        6.394E-08   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.370E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.061E-11
          TIME INCREMENT MAY NOW INCREASE TO    1.478E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  9.853E-04,  FRACTION OF STEP COMPLETED  1.478E-05
 STEP TIME COMPLETED       2.956E-03,  TOTAL TIME COMPLETED        0.163    


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.478E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      3.907E-02   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      3.907E-02
 LARGEST INCREMENT OF DISP.         1.674E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -4.687E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      2.154E-03   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.154E-03
 LARGEST INCREMENT OF DISP.         1.675E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        7.312E-08   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.590E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   6.956E-11
          TIME INCREMENT MAY NOW INCREASE TO    2.217E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.478E-03,  FRACTION OF STEP COMPLETED  2.217E-05
 STEP TIME COMPLETED       4.434E-03,  TOTAL TIME COMPLETED        0.164    


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.217E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      4.593E-02   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      4.593E-02
 LARGEST INCREMENT OF DISP.         1.962E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.495E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -3.205E-03   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -3.205E-03
 LARGEST INCREMENT OF DISP.         1.963E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        8.903E-08   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.905E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   9.562E-11
          TIME INCREMENT MAY NOW INCREASE TO    3.325E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.217E-03,  FRACTION OF STEP COMPLETED  3.325E-05
 STEP TIME COMPLETED       6.651E-03,  TOTAL TIME COMPLETED        0.167    


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.325E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      5.429E-02   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.429E-02
 LARGEST INCREMENT OF DISP.         2.301E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -6.442E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      2.936E-03   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.936E-03
 LARGEST INCREMENT OF DISP.         2.302E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.006E-07   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.217E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.314E-10
          TIME INCREMENT MAY NOW INCREASE TO    4.988E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.325E-03,  FRACTION OF STEP COMPLETED  4.988E-05
 STEP TIME COMPLETED       9.976E-03,  TOTAL TIME COMPLETED        0.170    


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.988E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      6.372E-02   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      6.372E-02
 LARGEST INCREMENT OF DISP.         2.697E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -7.552E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.809E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -4.390E-03   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.390E-03
 LARGEST INCREMENT OF DISP.         2.698E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.234E-07   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.637E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.807E-10
          TIME INCREMENT MAY NOW INCREASE TO    7.482E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.988E-03,  FRACTION OF STEP COMPLETED  7.482E-05
 STEP TIME COMPLETED       1.496E-02,  TOTAL TIME COMPLETED        0.175    


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.482E-03
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      7.522E-02   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      7.522E-02
 LARGEST INCREMENT OF DISP.         3.162E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -8.854E-05   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.810E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      4.031E-03   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      4.031E-03
 LARGEST INCREMENT OF DISP.         3.163E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.392E-07   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.061E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.483E-10
          TIME INCREMENT MAY NOW INCREASE TO    1.122E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.482E-03,  FRACTION OF STEP COMPLETED  1.122E-04
 STEP TIME COMPLETED       2.245E-02,  TOTAL TIME COMPLETED        0.182    


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.122E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      8.819E-02   AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      8.819E-02
 LARGEST INCREMENT OF DISP.         3.707E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.038E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.810E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -6.032E-03   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -6.032E-03
 LARGEST INCREMENT OF DISP.         3.709E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.704E-07   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.638E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.413E-10
          TIME INCREMENT MAY NOW INCREASE TO    1.683E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.122E-02,  FRACTION OF STEP COMPLETED  1.683E-04
 STEP TIME COMPLETED       3.367E-02,  TOTAL TIME COMPLETED        0.194    


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.683E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      0.104       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.104    
 LARGEST INCREMENT OF DISP.         4.346E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.217E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.810E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      5.538E-03   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.538E-03
 LARGEST INCREMENT OF DISP.         4.348E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.919E-07   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.214E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.692E-10
          TIME INCREMENT MAY NOW INCREASE TO    2.525E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.683E-02,  FRACTION OF STEP COMPLETED  2.525E-04
 STEP TIME COMPLETED       5.050E-02,  TOTAL TIME COMPLETED        0.211    


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.525E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      0.122       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.122    
 LARGEST INCREMENT OF DISP.         5.095E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.427E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.810E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -8.291E-03   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.291E-03
 LARGEST INCREMENT OF DISP.         5.098E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.405E-07   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.006E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   6.449E-10
          TIME INCREMENT MAY NOW INCREASE TO    3.788E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.525E-02,  FRACTION OF STEP COMPLETED  3.788E-04
 STEP TIME COMPLETED       7.575E-02,  TOTAL TIME COMPLETED        0.236    


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.788E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      0.144       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.144    
 LARGEST INCREMENT OF DISP.         5.974E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.672E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.810E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      7.610E-03   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      7.610E-03
 LARGEST INCREMENT OF DISP.         5.976E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.741E-07   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.793E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   8.865E-10
          TIME INCREMENT MAY NOW INCREASE TO    5.682E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.788E-02,  FRACTION OF STEP COMPLETED  5.682E-04
 STEP TIME COMPLETED       0.114    ,  TOTAL TIME COMPLETED        0.274    


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.682E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      0.168       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.168    
 LARGEST INCREMENT OF DISP.         7.004E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.961E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.810E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -1.140E-02   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.140E-02
 LARGEST INCREMENT OF DISP.         7.007E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.236E-07   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.875E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.219E-09
          TIME INCREMENT MAY NOW INCREASE TO    8.522E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.682E-02,  FRACTION OF STEP COMPLETED  8.522E-04
 STEP TIME COMPLETED       0.170    ,  TOTAL TIME COMPLETED        0.330    


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.522E-02
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      0.198       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.198    
 LARGEST INCREMENT OF DISP.         8.212E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.299E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.810E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      1.046E-02   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.046E-02
 LARGEST INCREMENT OF DISP.         8.215E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.896E-07   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.951E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.675E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.128    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.522E-02,  FRACTION OF STEP COMPLETED  1.278E-03
 STEP TIME COMPLETED       0.256    ,  TOTAL TIME COMPLETED        0.416    


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.128    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      0.232       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.232    
 LARGEST INCREMENT OF DISP.         9.628E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.695E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.810E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -1.567E-02   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -1.567E-02
 LARGEST INCREMENT OF DISP.         9.632E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        4.778E-07   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.428E-40   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.303E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.192    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.128    ,  FRACTION OF STEP COMPLETED  1.918E-03
 STEP TIME COMPLETED       0.384    ,  TOTAL TIME COMPLETED        0.544    


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.192    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      0.273       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.273    
 LARGEST INCREMENT OF DISP.         1.129E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.159E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.810E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      1.436E-02   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.436E-02
 LARGEST INCREMENT OF DISP.         1.129E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        5.176E-07   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.088E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   3.169E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.288    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.192    ,  FRACTION OF STEP COMPLETED  2.876E-03
 STEP TIME COMPLETED       0.575    ,  TOTAL TIME COMPLETED        0.735    


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.288    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      0.320       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.320    
 LARGEST INCREMENT OF DISP.         1.324E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -3.704E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.811E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  9ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -2.153E-02   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.153E-02
 LARGEST INCREMENT OF DISP.         1.324E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        6.320E-07   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.289E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.360E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.431    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.288    ,  FRACTION OF STEP COMPLETED  4.314E-03
 STEP TIME COMPLETED       0.863    ,  TOTAL TIME COMPLETED         1.02    


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.431    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      0.377       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.377    
 LARGEST INCREMENT OF DISP.         1.552E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -4.342E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.811E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  8ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      1.986E-02   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      1.986E-02
 LARGEST INCREMENT OF DISP.         1.553E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        6.895E-07   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.490E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.996E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.647    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.431    ,  FRACTION OF STEP COMPLETED  6.472E-03
 STEP TIME COMPLETED        1.29    ,  TOTAL TIME COMPLETED         1.45    


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.647    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE      0.440       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.440    
 LARGEST INCREMENT OF DISP.         1.820E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.091E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.811E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE     -2.989E-02   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -2.989E-02
 LARGEST INCREMENT OF DISP.         1.821E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        8.498E-07   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.762E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.755E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   8.251E-09
          TIME INCREMENT MAY NOW INCREASE TO    0.971    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.647    ,  FRACTION OF STEP COMPLETED  9.707E-03
 STEP TIME COMPLETED        1.94    ,  TOTAL TIME COMPLETED         2.10    


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.971    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE      0.518       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.518    
 LARGEST INCREMENT OF DISP.         2.134E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -5.971E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.763E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.811E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE      2.741E-02   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      2.741E-02
 LARGEST INCREMENT OF DISP.         2.135E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.056E-06   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.763E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.024E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.135E-08
          TIME INCREMENT MAY NOW INCREASE TO     1.46    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.971    ,  FRACTION OF STEP COMPLETED  1.456E-02
 STEP TIME COMPLETED        2.91    ,  TOTAL TIME COMPLETED         3.07    


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   1.46    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE      0.607       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.607    
 LARGEST INCREMENT OF DISP.         2.503E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -7.000E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.763E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.812E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE     -4.117E-02   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -4.117E-02
 LARGEST INCREMENT OF DISP.         2.504E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.245E-06   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.763E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.387E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.561E-08
          TIME INCREMENT MAY NOW INCREASE TO     2.18    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   1.46    ,  FRACTION OF STEP COMPLETED  2.184E-02
 STEP TIME COMPLETED        4.37    ,  TOTAL TIME COMPLETED         4.53    


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   2.18    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE      0.715       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.715    
 LARGEST INCREMENT OF DISP.         2.936E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -8.206E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.763E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.812E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE      3.761E-02   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      3.761E-02
 LARGEST INCREMENT OF DISP.         2.937E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.421E-06   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.763E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.750E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.149E-08
          TIME INCREMENT MAY NOW INCREASE TO     3.28    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   2.18    ,  FRACTION OF STEP COMPLETED  3.276E-02
 STEP TIME COMPLETED        6.55    ,  TOTAL TIME COMPLETED         6.71    


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   3.28    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE      0.837       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.837    
 LARGEST INCREMENT OF DISP.         3.443E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -9.622E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.763E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.813E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE     -5.661E-02   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -5.661E-02
 LARGEST INCREMENT OF DISP.         3.445E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.728E-06   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.763E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.232E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.960E-08
          TIME INCREMENT MAY NOW INCREASE TO     4.91    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   3.28    ,  FRACTION OF STEP COMPLETED  4.914E-02
 STEP TIME COMPLETED        9.83    ,  TOTAL TIME COMPLETED         9.99    


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   4.91    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       317.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE      0.986       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.986    
 LARGEST INCREMENT OF DISP.         4.039E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.128E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.763E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.813E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE      5.174E-02   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      5.174E-02
 LARGEST INCREMENT OF DISP.         4.041E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.097E-06   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.763E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.720E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   4.059E-08
          TIME INCREMENT MAY NOW INCREASE TO     7.37    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   4.91    ,  FRACTION OF STEP COMPLETED  7.372E-02
 STEP TIME COMPLETED        14.7    ,  TOTAL TIME COMPLETED         14.9    


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   7.37    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE       1.15       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       1.15    
 LARGEST INCREMENT OF DISP.         4.737E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.324E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.764E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.814E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       317.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE     -7.813E-02   AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -7.813E-02
 LARGEST INCREMENT OF DISP.         4.740E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.568E-06   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.764E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.353E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   5.591E-08
          TIME INCREMENT MAY NOW INCREASE TO     11.1    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   7.37    ,  FRACTION OF STEP COMPLETED  0.111    
 STEP TIME COMPLETED        22.1    ,  TOTAL TIME COMPLETED         22.3    


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   11.1    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE       1.36       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       1.36    
 LARGEST INCREMENT OF DISP.         5.551E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.560E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.764E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.815E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE      7.100E-02   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      7.100E-02
 LARGEST INCREMENT OF DISP.         5.562E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        1.127E-05   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.764E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.008E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   7.708E-08
          TIME INCREMENT MAY NOW INCREASE TO     16.6    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   11.1    ,  FRACTION OF STEP COMPLETED  0.166    
 STEP TIME COMPLETED        33.2    ,  TOTAL TIME COMPLETED         33.3    


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   16.6    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE       1.60       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       1.60    
 LARGEST INCREMENT OF DISP.         6.532E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -1.810E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.764E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.816E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE     -0.108       AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.108    
 LARGEST INCREMENT OF DISP.         6.535E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.121E-06   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.764E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.829E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.066E-07
          TIME INCREMENT MAY NOW INCREASE TO     24.9    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   16.6    ,  FRACTION OF STEP COMPLETED  0.249    
 STEP TIME COMPLETED        49.8    ,  TOTAL TIME COMPLETED         49.9    


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   24.9    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  7ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       316.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE       1.88       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       1.88    
 LARGEST INCREMENT OF DISP.         7.677E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.126E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.765E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.817E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       316.       TIME AVG. FORCE        317.    
 LARGEST SCALED RESIDUAL FORCE      9.763E-02   AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      9.763E-02
 LARGEST INCREMENT OF DISP.         7.680E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        2.880E-06   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.765E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.687E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.455E-07
          TIME INCREMENT MAY NOW INCREASE TO     37.3    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   24.9    ,  FRACTION OF STEP COMPLETED  0.373    
 STEP TIME COMPLETED        74.6    ,  TOTAL TIME COMPLETED         74.8    


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   37.3    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       315.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE       2.20       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       2.20    
 LARGEST INCREMENT OF DISP.         9.011E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.508E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.765E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.818E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       315.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -0.150       AT NODE        365   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -0.150    
 LARGEST INCREMENT OF DISP.         9.014E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        3.400E-06   AT NODE         83   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.765E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.646E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   1.998E-07
          TIME INCREMENT MAY NOW INCREASE TO     56.0    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   37.3    ,  FRACTION OF STEP COMPLETED  0.560    
 STEP TIME COMPLETED        112.    ,  TOTAL TIME COMPLETED         112.    


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   56.0    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       313.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE       2.60       AT NODE         22   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE       2.60    
 LARGEST INCREMENT OF DISP.         1.058E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.943E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.766E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.820E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  5ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       313.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      0.131       AT NODE        374   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.131    
 LARGEST INCREMENT OF DISP.         1.058E-02   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.        4.023E-06   AT NODE        375   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.766E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.762E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   2.753E-07

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   56.0    ,  FRACTION OF STEP COMPLETED  0.840    
 STEP TIME COMPLETED        168.    ,  TOTAL TIME COMPLETED         168.    


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   32.1    
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       313.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE      0.287       AT NODE        368   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE      0.287    
 LARGEST INCREMENT OF DISP.         5.110E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -9.521E-04   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.766E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -7.544E-33   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  3023
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.852e+06
        SOLVER ELAPSED TIME:  6ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       313.       TIME AVG. FORCE        316.    
 LARGEST SCALED RESIDUAL FORCE     -8.849E-05   AT NODE        367   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
  CORRESPONDING RESIDUAL FORCE     -8.849E-05
 LARGEST INCREMENT OF DISP.         5.110E-03   AT NODE          1   DOF  2
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO DISP.       -2.187E-07   AT NODE        111   DOF  2
   INSTANCE: ELG Specimen 2024-1                                                             
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       147.    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION     -1.766E-32   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.231E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 MAXIMUM DIFFERENCE IN CREEP STRAIN INCREMENT IS   6.423E-08

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   32.1    ,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED        200.    ,  TOTAL TIME COMPLETED         200.    


          THE ANALYSIS HAS BEEN COMPLETED



     ANALYSIS SUMMARY:
     TOTAL OF         49  INCREMENTS
                       0  CUTBACKS IN AUTOMATIC INCREMENTATION
                     103  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     103  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                     103  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       3  WARNING MESSAGES DURING USER INPUT PROCESSING
                       4  WARNING MESSAGES DURING ANALYSIS
                       0  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                       0  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =      3.2    
       SYSTEM TIME (SEC)    =     0.40    
       TOTAL CPU TIME (SEC) =      3.6    
       WALLCLOCK TIME (SEC) =           11
       MEMORY PEAK (GB)     =            0
