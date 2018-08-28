  /* 
  * Author: Jordan Micah Bennett
  * Topic?: Basic artificial neural network re-written using matrix notation
  * 
  * The problem space is X-OR inputs. So the model does xor input prediction.
  * Given two numbers in X-OR space, the model will try to guess the output.
  * 
  * For:
  * a) Input=(0,0) output should be 0
  * b) Input=(1,0) output should be 1
  * c) Input=(0,1) output should be 1
  * d) Input=(1,1) output should be 0
  * 
  */
 
import java.util.ArrayList;
import java.util.Scanner;


public class NeuralNetwork_xOR_MatrixVersion_Execution
{
    /*
     *      LAYER-I                              LAYER-II                                   LAYER-III
     *      
     *      neuron-A        weight-from-A        neuron-B        weight-to-G-from-B         neuron-G  
     *      neuron-C        weight-from-C        neuron-D        weight-to-G-from-D        
     *      neuron-E        weight-from-E        neuron-F        weight-to-G-from-F    
     */
    
    //problem space
    static ArrayList <String> problemSpace = getProblemSpace ( );
    
    //hyperparams
    static double alpha = 0.5; //momentum
    static double mse = 0.0; //mean squared error
    static double eta = 0.2; //learning rate
    static double bias = 1.0; //helps model to generate good answers, beyond the origin (0,0) on cartesian plane.

            //consumes 8 outcomes per training step (each outcome represents a neuron's output value)
    static Matrix LayerI_Outcomes = new Matrix ( 3, 1 ); //where last column entry represents bias neuron value
    static Matrix LayerII_Outcomes = new Matrix ( 3, 1 );
    static Matrix LayerIII_Outcomes = new Matrix ( 1, 1 ); 
    
            //consumes 3 gradients per training step (each gradient represents a neuron's gradient value)
    static Matrix LayerII_Gradients = new Matrix ( 2, 1 );
    static Matrix LayerIII_Gradients = new Matrix ( 1, 1 ); 
    
            //consumes 6 weights per training step
    static Matrix LayerI_II_Weights_FromNeuronA = new Matrix ( 2, 1 );
    static Matrix LayerI_II_Weights_FromNeuronC = new Matrix ( 2, 1 );
    static Matrix LayerI_II_Weights_FromNeuronE = new Matrix ( 2, 1 );
    static Matrix LayerII_III_Weights_ToNeuronG = new Matrix ( 3, 1 ); 
    
    static Matrix LayerI_II_DeltaWeights_FromNeuronA = new Matrix ( 2, 1 );
    static Matrix LayerI_II_DeltaWeights_FromNeuronC = new Matrix ( 2, 1 );
    static Matrix LayerI_II_DeltaWeights_FromNeuronE = new Matrix ( 2, 1 ); 
    static Matrix LayerII_III_DeltaWeights_ToNeuronG = new Matrix ( 3, 1 );       
             
            
    public static void main ( String [ ] arguments )
    {
        /////////////////////////
        //NEURAL NETWORK TRAINING
            /////////////////define starting conditions
            //System.out.println ( "\t\t__BIAS NEURON SETUP__" );
            LayerI_Outcomes.getMatrix ( ) [ 2 ] [ 0 ] = bias; //these matrix entries remain constant over time
            LayerII_Outcomes.getMatrix ( ) [ 2 ] [ 0 ] = bias;

            //System.out.println ( "__WEIGHT & GRADIENT SETUP__" );
            
            //System.out.println ( "\t\t__WEIGHT-SETUP" );
    /*
            LayerI_II_Weights_FromNeuronA.randomize ( );
            LayerI_II_Weights_FromNeuronC.randomize ( );                
            LayerI_II_Weights_FromNeuronE.randomize ( );
            LayerII_III_Weights_ToNeuronG.randomize ( );*/
        
        /*
        Weights are now initialized with good starting values, instead of random values.

        1) The model was ran, and good starting weights were observed.

        2) Starting weights are just the values of all the weights after initial problem space is consumed, that is the value of the weights after the first training case causes weights to obtain some random value between 0 and 1.

        3) Good starting weights are those that crucially those that lead to good hypotheses.
        
        4) So I ran the model a few times, and took weights that produced good hypotheses/results, then I copied those good starting weights such that the model now starts with weights initialized as those values.
         */
        
            //Instead of starting with random weights, I start instead with a set of good starting weights generated particularly when the model performs 
            //well.
        LayerI_II_Weights_FromNeuronA.setColumnMatrix 
        ( 
            new double [ ] 
            {
                0.5330270559252573,
                0.3770145997674229
            }
        );
            
                LayerI_II_Weights_FromNeuronC.setColumnMatrix 
        ( 
            new double [ ] 
            {
                0.384619730589013,
                0.3183092287291529
            }
        );
            
        LayerI_II_Weights_FromNeuronE.setColumnMatrix 
        ( 
            new double [ ] 
            {
                0.6014823191150609,
                0.4511163515443918
            }
        );
            
        LayerII_III_Weights_ToNeuronG.setColumnMatrix 
        ( 
            new double [ ] 
            {
                0.4863195706057799,
                0.808264485311589,
                0.30919672657537034
            }
        );
            
            //System.out.println ( "\t\t__GRADIENT-SETUP" );
            LayerII_Gradients.setMatrix ( 0 );
            LayerIII_Gradients.setMatrix ( 0 );
      
            trainModel ( );
            renderMenu ( );
    }
    
    public static void trainModel ( )
    {
        /////////////////////////
        //Run training - Extract training data, and run forward and backward propagation
        //Each step in loop consumes two inputs and an expected outcome
        
        for ( int pSI = 0; pSI < problemSpace.size ( ); pSI ++ )
        {
            String [ ] trainingData = problemSpace.get ( pSI ).split ( ":" );
            
            String [ ] inputSpace = trainingData [ 0 ].split ( "," );
            //System.out.println ( "\n\nPROBLEM SPACE --> " + pSI );
            /////////////////////////////////////////////////////////////////////////////////////
            /////////////////FORWARD PASS
            //System.out.println ( "__BUILD-FORWARD-PASS__[" + pSI + "]" );
            forwardPass ( inputSpace );
            
            /////////////////////////////////////////////////////////////////////////////////////
            /////////////////BACKWARD PASS
            //System.out.println ( "__BUILD-BACKWARD-PASS__[" + pSI + "]" );
            double targetData = Double.parseDouble ( problemSpace.get ( pSI ).split ( ":" ) [ 1 ] ); //just the associated target value per training input case
        
            backwardPass ( targetData );
            //System.out.println ( "END-OF-CYCLE [" + pSI + "]\n\n\n" );
        } 
        System.out.println ( "MEAN SQUARED ERROR  --> " + mse );
    }
    
    public static ArrayList getProblemSpace ( )
    {
        //PROBLEM SPACE
        ArrayList <String> returnValue = new ArrayList <String> ( );

        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );   
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );        
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
        returnValue.add ( "1,0:1" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "0,1:1" );
        returnValue.add ( "0,0:0" );
        returnValue.add ( "1,1:0" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );   
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );
		returnValue.add ( "1,1:0" );
		returnValue.add ( "1,0:1" );
		returnValue.add ( "0,1:1" );
		returnValue.add ( "0,0:0" );


        return returnValue;
    }
    
    public static void forwardPass ( String [ ] inputSpace ) //pSI = problem space iterator
    {
        //consumes 2 inputs per training step
        //System.out.println ( "__BUILD-FORWARD-PASS__[" + pSI + "]" );
        LayerI_Outcomes.setColumnMatrix ( new double [ ] { Double.parseDouble ( inputSpace [ 0 ] ), Double.parseDouble ( inputSpace [ 1 ] ) } );
            
        //zeroethWeightsFromLayer0 represents 0'th weights stemming from each neuron in layer 0, going to neuron B. 
        //The aim is to take each weight, multiply it by each relevant layer 0 outcome, then add each product to an accumulating sum.
        //The resulting 'productsOfPriorWeightsAndPriorOutcomesToLayerIINeuronB' matrix contains three entries, each representing an accumulation of prior products.
        //The layer II's outcomes is then built in terms of the the last entry in 
            Matrix zeroethWeightsFromLayer0 = new Matrix ( 3, 1 ); 
            zeroethWeightsFromLayer0.setColumnMatrix ( new double [ ] { LayerI_II_Weights_FromNeuronA.getMatrix ( ) [ 0 ] [ 0 ], LayerI_II_Weights_FromNeuronC.getMatrix ( ) [ 0 ] [ 0 ], LayerI_II_Weights_FromNeuronE.getMatrix ( ) [ 0 ] [ 0 ] } );
            Matrix productsOfPriorWeightsAndPriorOutcomesToLayerIINeuronB = zeroethWeightsFromLayer0.getAccumulatedDotProductWithoutSingularSum ( LayerI_Outcomes );
        
        //onethWeightsFromLayer0 represents 1'th weights stemming from each neuron in layer 0, going to neuron D. 
        //The aim is to take each weight, multiply it by each relevant layer 0 outcome, then add each product to an accumulating sum.
        //The resulting 'productsOfPriorWeightsAndPriorOutcomesToLayerIINeuronD' matrix contains three entries, each representing an accumulation of prior products.
            Matrix onethWeightsFromLayer0 = new Matrix ( 3, 1 ); 
            onethWeightsFromLayer0.setColumnMatrix ( new double [ ] { LayerI_II_Weights_FromNeuronA.getMatrix ( ) [ 1 ] [ 0 ], LayerI_II_Weights_FromNeuronC.getMatrix ( ) [ 1 ] [ 0 ], LayerI_II_Weights_FromNeuronE.getMatrix ( ) [ 1 ] [ 0 ] } );
            Matrix productsOfPriorWeightsAndPriorOutcomesToLayerIINeuronD = onethWeightsFromLayer0.getAccumulatedDotProductWithoutSingularSum ( LayerI_Outcomes );
       
        //The layer II's outcomes is then built in terms of the the last entry in zeroethWeightsFromLayer0 and onethWeightsFromLayer0.
        //LayerII's outcome signifies the final accumulation from the process.
        //Recall: Forward Prop ---> sigma += priorWeight * priorOutcome
        
            //redefine layer 2 outcomes in terms of operations on weights and neuron outcomes of prior later
            //System.out.println ( "\t\t__BUILD-FORWARD-PASS/BUILD-ACTIVATION-ON-SUM-OF-PRODUCTS--LAYER_2" );
            LayerII_Outcomes.setColumnMatrix ( new double [ ] { productsOfPriorWeightsAndPriorOutcomesToLayerIINeuronB.getMatrix ( ) [ 2 ][ 0 ], productsOfPriorWeightsAndPriorOutcomesToLayerIINeuronD.getMatrix ( ) [ 2 ][ 0 ] } ); //sumOfSumOfProductOfPriorWeightsAndPriorOutcomesToLayerIINeuronB, sumOfSumOfProductOfPriorWeightsAndPriorOutcomesToLayerIINeuronD
            LayerII_Outcomes.setColumnMatrix ( new double [ ] { LayerII_Outcomes.getActivation ( ).getMatrix ( ) [ 0 ] [ 0 ], LayerII_Outcomes.getActivation ( ).getMatrix ( ) [ 1 ] [ 0 ] }  ); //prime activation on each entry, where each entry is a sum
            
            //consume layer 2 outcomes and relevant weights, by producing a sum of products
            //System.out.println ( "\t\t__BUILD-FORWARD-PASS/BUILD-SUM-OF-PRODUCTS--LAYER_2__" );
            Matrix zeroethWeightsFromLayer1 = new Matrix ( 3, 1 ); 
            zeroethWeightsFromLayer1.setColumnMatrix ( new double [ ] { LayerII_III_Weights_ToNeuronG.getMatrix ( ) [ 1 ] [ 0 ], LayerII_III_Weights_ToNeuronG.getMatrix ( ) [ 1 ] [ 0 ], LayerII_III_Weights_ToNeuronG.getMatrix ( ) [ 1 ] [ 0 ] } );
            Matrix productsOfPriorWeightsAndPriorOutcomesToLayerIIINeuronG = zeroethWeightsFromLayer1.getAccumulatedDotProductWithoutSingularSum ( LayerII_Outcomes );

            //redefine layer 3 outcomes in terms of operations on weights and neuron outcomes of prior later
            //System.out.println ( "\t\t__BUILD-FORWARD-PASS/BUILD-ACTIVATION-ON-SUM-OF-PRODUCTS--LAYER_3__" );
            LayerIII_Outcomes.setColumnMatrix ( new double [ ] { productsOfPriorWeightsAndPriorOutcomesToLayerIIINeuronG.getMatrix ( ) [ 2 ][ 0 ] } );  
            
            //activate layer III value with tanh
            LayerIII_Outcomes.setColumnMatrix ( new double [ ] { LayerIII_Outcomes.getActivation ( ).getMatrix ( ) [ 0 ] [ 0 ] } ); //activation on each entry, where each entry is a sum         
    }
        
    public static void forwardPass ( Matrix testInputs ) 
    {
        LayerI_Outcomes.setMatrix ( testInputs ); //setMatrix is okay here, since it resets up to the parameter size, so bias neuron value remains constant
        //consumes 2 inputs per training step
      
        //zeroethWeightsFromLayer0 represents 0'th weights stemming from each neuron in layer 0, going to neuron B. 
        //The aim is to take each weight, multiply it by each relevant layer 0 outcome, then add each product to an accumulating sum.
        //The resulting 'productsOfPriorWeightsAndPriorOutcomesToLayerIINeuronB' matrix contains three entries, each representing an accumulation of prior products.
        //The layer II's outcomes is then built in terms of the the last entry in 
            Matrix zeroethWeightsFromLayer0 = new Matrix ( 3, 1 ); 
            zeroethWeightsFromLayer0.setColumnMatrix ( new double [ ] { LayerI_II_Weights_FromNeuronA.getMatrix ( ) [ 0 ] [ 0 ], LayerI_II_Weights_FromNeuronC.getMatrix ( ) [ 0 ] [ 0 ], LayerI_II_Weights_FromNeuronE.getMatrix ( ) [ 0 ] [ 0 ] } );
            Matrix productsOfPriorWeightsAndPriorOutcomesToLayerIINeuronB = zeroethWeightsFromLayer0.getAccumulatedDotProductWithoutSingularSum ( LayerI_Outcomes );
        
        //onethWeightsFromLayer0 represents 1'th weights stemming from each neuron in layer 0, going to neuron D. 
        //The aim is to take each weight, multiply it by each relevant layer 0 outcome, then add each product to an accumulating sum.
        //The resulting 'productsOfPriorWeightsAndPriorOutcomesToLayerIINeuronD' matrix contains three entries, each representing an accumulation of prior products.
            Matrix onethWeightsFromLayer0 = new Matrix ( 3, 1 ); 
            onethWeightsFromLayer0.setColumnMatrix ( new double [ ] { LayerI_II_Weights_FromNeuronA.getMatrix ( ) [ 1 ] [ 0 ], LayerI_II_Weights_FromNeuronC.getMatrix ( ) [ 1 ] [ 0 ], LayerI_II_Weights_FromNeuronE.getMatrix ( ) [ 1 ] [ 0 ] } );
            Matrix productsOfPriorWeightsAndPriorOutcomesToLayerIINeuronD = onethWeightsFromLayer0.getAccumulatedDotProductWithoutSingularSum ( LayerI_Outcomes );
       
        //The layer II's outcomes is then built in terms of the the last entry in zeroethWeightsFromLayer0 and onethWeightsFromLayer0.
        //LayerII's outcome signifies the final accumulation from the process.
        //Recall: Forward Prop ---> sigma += priorWeight * priorOutcome
        
            //redefine layer 2 outcomes in terms of operations on weights and neuron outcomes of prior later
            //System.out.println ( "\t\t__BUILD-FORWARD-PASS/BUILD-ACTIVATION-ON-SUM-OF-PRODUCTS--LAYER_2" );
            LayerII_Outcomes.setColumnMatrix ( new double [ ] { productsOfPriorWeightsAndPriorOutcomesToLayerIINeuronB.getMatrix ( ) [ 2 ][ 0 ], productsOfPriorWeightsAndPriorOutcomesToLayerIINeuronD.getMatrix ( ) [ 2 ][ 0 ] } ); //sumOfSumOfProductOfPriorWeightsAndPriorOutcomesToLayerIINeuronB, sumOfSumOfProductOfPriorWeightsAndPriorOutcomesToLayerIINeuronD
            LayerII_Outcomes.setColumnMatrix ( new double [ ] { LayerII_Outcomes.getActivation ( ).getMatrix ( ) [ 0 ] [ 0 ], LayerII_Outcomes.getActivation ( ).getMatrix ( ) [ 1 ] [ 0 ] }  ); //prime activation on each entry, where each entry is a sum
            
            //consume layer 2 outcomes and relevant weights, by producing a sum of products
            //System.out.println ( "\t\t__BUILD-FORWARD-PASS/BUILD-SUM-OF-PRODUCTS--LAYER_2__" );
            Matrix zeroethWeightsFromLayer1 = new Matrix ( 3, 1 ); 
            zeroethWeightsFromLayer1.setColumnMatrix ( new double [ ] { LayerII_III_Weights_ToNeuronG.getMatrix ( ) [ 1 ] [ 0 ], LayerII_III_Weights_ToNeuronG.getMatrix ( ) [ 1 ] [ 0 ], LayerII_III_Weights_ToNeuronG.getMatrix ( ) [ 1 ] [ 0 ] } );
            Matrix productsOfPriorWeightsAndPriorOutcomesToLayerIIINeuronG = zeroethWeightsFromLayer1.getAccumulatedDotProductWithoutSingularSum ( LayerII_Outcomes );

            //redefine layer 3 outcomes in terms of operations on weights and neuron outcomes of prior later
            //System.out.println ( "\t\t__BUILD-FORWARD-PASS/BUILD-ACTIVATION-ON-SUM-OF-PRODUCTS--LAYER_3__" );
            LayerIII_Outcomes.setColumnMatrix ( new double [ ] { productsOfPriorWeightsAndPriorOutcomesToLayerIIINeuronG.getMatrix ( ) [ 2 ][ 0 ] } );  
            
            //activate layer III value with tanh
            LayerIII_Outcomes.setColumnMatrix ( new double [ ] { LayerIII_Outcomes.getActivation ( ).getMatrix ( ) [ 0 ] [ 0 ] } ); //activation on each entry, where each entry is a sum         
    }
    
    public static void backwardPass ( double targetData )
    {
            //error computation
            double errorSigma = 0;
            
            errorSigma += Math.pow ( targetData - LayerIII_Outcomes.getMatrix ( ) [ 0 ] [ 0 ], 2 );
    
            mse = errorSigma / 2;   //mean squared error
        
            //consume layer 3 outcomes and gradients by producing an output (final layer) gradient
            //System.out.println ( "\t\t__BUILD-BACKWARD-PASS/BUILD-OUTCOME-GRADIENT-OPERATION__" );
            
            double deltaOnOutcomeNeuron = targetData - LayerIII_Outcomes.getMatrix ( ) [ 0 ] [ 0 ]; //delta, where layerIII represents output/final layer
            
            double productOfDeltaAndPrimeActivationOnOutcome = deltaOnOutcomeNeuron * LayerIII_Outcomes.getPrimeActivation ( ).getMatrix ( ) [ 0 ] [ 0 ]; //delta * primeActivation ( currentOutcome )
            
            LayerIII_Gradients.setColumnMatrix ( new double [ ] { productOfDeltaAndPrimeActivationOnOutcome } ); //update output/final layer gradients
            
            //consume layer 2 weights and layer 3 gradients, by producing hidden gradients.
            //the term "layer 2" above is not to be mistaken for belonging only to layer 2, but to be taken to relate to come from neurons in layer 2.
            //System.out.println ( "\t\t__BUILD-BACKWARD-PASS/BUILD-HIDDEN-GRADIENT-OPERATION__" );
            //Formula: hiddenLayerSigma * primeActivation(currentOutcome), where hiddenLayerSigma: thisWeight*nextGradient
            //The following block builds each part of the formula above in terms of matrices, then combines them to form the input of the gradient
            Matrix hiddenLayerSigma = new Matrix ( 2, 1 );
            hiddenLayerSigma.setColumnMatrix 
                    ( 
                         new double [ ] 
                         {
                             LayerII_III_Weights_ToNeuronG.getProduct ( LayerIII_Gradients.getMatrix ( ) [ 0 ] [ 0 ] ).getMatrix ( ) [ 0 ] [ 0 ],
                             LayerII_III_Weights_ToNeuronG.getProduct ( LayerIII_Gradients.getMatrix ( ) [ 0 ] [ 0 ] ).getMatrix ( ) [ 1 ] [ 0 ]
                         }
                    ); //gradient. Focus is on element 0 of output gradients, because only one neuron exists conceptually in layer 3, that's not bias based (index 2).       
 
            Matrix sigmaMultiples = new Matrix ( 2, 1 );
            sigmaMultiples.setColumnMatrix 
                    ( 
                         new double [ ] 
                         {
                             LayerII_Outcomes.getPrimeActivation ( ).getMatrix ( ) [ 0 ] [ 0 ],
                             LayerII_Outcomes.getPrimeActivation ( ).getMatrix ( ) [ 1 ] [ 0 ]
                         }
                    ); 
                    
            Matrix productOfHiddenLayerSigmaAndPrimeActivation = sigmaMultiples.getProduct ( hiddenLayerSigma );

            LayerII_Gradients.setMatrix ( productOfHiddenLayerSigmaAndPrimeActivation ); //update hidden/middle layer gradients ...

                    
            //consume layer 2 and 3 structures, by producing updated weights/delta weights   
            //System.out.println ( "\t\t__BUILD-BACKWARD-PASS/BUILD-WEIGHT-UPDATE-OPERATION__" );
                //Layer 2 ops
                 //new delta weight --> ((eta*thisGradient*priorOutcome) + (alpha*oldDelta))
                 //The block below builds "((eta*thisGradient*priorOutcome)" in matrix terms, then "(alpha*oldDelta))" in matrix terms, 
                 //then summate them to form a separate matrix.
                    //WRT neuron A and B/D
                    Matrix oldWeightsFromAtoBandD = new Matrix ( 2, 1 );
                    oldWeightsFromAtoBandD.setColumnMatrix ( new double [ ] { LayerI_II_Weights_FromNeuronA.getMatrix ( ) [ 0 ] [ 0 ], LayerI_II_Weights_FromNeuronA.getMatrix ( ) [ 1 ] [ 0 ] } );
                    
                    Matrix oldDeltaWeightsFromAtoBandD = new Matrix ( 2, 1 );
                    oldDeltaWeightsFromAtoBandD.setColumnMatrix ( new double [ ] { LayerI_II_DeltaWeights_FromNeuronA.getMatrix ( ) [ 0 ] [ 0 ], LayerI_II_DeltaWeights_FromNeuronA.getMatrix ( ) [ 1 ] [ 0 ] } );
                    
                    Matrix newDeltaWeightsTermI_etaProduct_wrtNeuronA = LayerII_Gradients.getProduct ( LayerI_Outcomes.getMatrix ( ) [ 0 ] [ 0 ] ).getProduct ( eta ); //(eta*thisGradient*priorOutcomeFromLayerIIWrtNeuronA)
                    //Only two gradients in hidden layer. This 2 sized matrix will multiply nicely by the 'newDeltaWeightTermII_alphaProduct' term.
                    
                    Matrix newDeltaWeightsTermII_alphaProduct_wrtNeuronA = oldDeltaWeightsFromAtoBandD.getProduct ( alpha ); //apha*oldDeltas
                    
                    Matrix newDeltaWeights_WrtNeuronAandD = newDeltaWeightsTermI_etaProduct_wrtNeuronA.getSum ( newDeltaWeightsTermII_alphaProduct_wrtNeuronA );
                    /////////////////////////
                    //update weights
                    LayerI_II_Weights_FromNeuronA.setMatrix ( LayerI_II_Weights_FromNeuronA.getSum ( newDeltaWeights_WrtNeuronAandD ) );
                    LayerI_II_DeltaWeights_FromNeuronA.setMatrix ( newDeltaWeights_WrtNeuronAandD );
                    
                    
                    
                    
                    
                    //WRT neuron C and B/D
                    Matrix oldWeightsFromCtoBandD = new Matrix ( 2, 1 );
                    oldWeightsFromCtoBandD.setColumnMatrix ( new double [ ] { LayerI_II_Weights_FromNeuronC.getMatrix ( ) [ 0 ] [ 0 ], LayerI_II_Weights_FromNeuronC.getMatrix ( ) [ 1 ] [ 0 ] } );
                    
                    Matrix oldDeltaWeightsFromCtoBandD = new Matrix ( 2, 1 );
                    oldDeltaWeightsFromCtoBandD.setColumnMatrix ( new double [ ] { LayerI_II_DeltaWeights_FromNeuronC.getMatrix ( ) [ 0 ] [ 0 ], LayerI_II_DeltaWeights_FromNeuronC.getMatrix ( ) [ 1 ] [ 0 ] } );
                    
                    Matrix newDeltaWeightsTermI_etaProduct_wrtNeuronC = LayerII_Gradients.getProduct ( LayerI_Outcomes.getMatrix ( ) [ 1 ] [ 0 ] ).getProduct ( eta ); //(eta*thisGradient*priorOutcomeFromLayerIIWrtNeuronC)
                    //Only two gradients in hidden layer. This 2 sized matrix will multiply nicely by the 'newDeltaWeightTermII_alphaProduct' term.
                    
                    Matrix newDeltaWeightsTermII_alphaProduct_wrtNeuronC = oldDeltaWeightsFromCtoBandD.getProduct ( alpha ); //apha*oldDeltas
                    
                    Matrix newDeltaWeights_WrtNeuronCandD = newDeltaWeightsTermI_etaProduct_wrtNeuronC.getSum ( newDeltaWeightsTermII_alphaProduct_wrtNeuronC );
                    /////////////////////////
                    //update weights
                    LayerI_II_Weights_FromNeuronC.setMatrix ( LayerI_II_Weights_FromNeuronC.getSum ( newDeltaWeights_WrtNeuronCandD ) );
                    LayerI_II_DeltaWeights_FromNeuronC.setMatrix ( newDeltaWeights_WrtNeuronCandD );
                    
                    
                    
                    
                    
                    //WRT neuron E and B/D
                    Matrix oldWeightsFromEtoBandD = new Matrix ( 2, 1 );
                    oldWeightsFromEtoBandD.setColumnMatrix ( new double [ ] { LayerI_II_Weights_FromNeuronE.getMatrix ( ) [ 0 ] [ 0 ], LayerI_II_Weights_FromNeuronE.getMatrix ( ) [ 1 ] [ 0 ] } );
                    
                    Matrix oldDeltaWeightsFromEtoBandD = new Matrix ( 2, 1 );
                    oldDeltaWeightsFromEtoBandD.setColumnMatrix ( new double [ ] { LayerI_II_DeltaWeights_FromNeuronE.getMatrix ( ) [ 0 ] [ 0 ], LayerI_II_DeltaWeights_FromNeuronE.getMatrix ( ) [ 1 ] [ 0 ] } );
                    
                    Matrix newDeltaWeightsTermI_etaProduct_wrtNeuronE = LayerII_Gradients.getProduct ( LayerI_Outcomes.getMatrix ( ) [ 2 ] [ 0 ] ).getProduct ( eta ); //(eta*thisGradient*priorOutcomeFromLayerIIWrtNeuronE)
                    //Only two gradients in hidden layer. This 2 sized matrix will multiply nicely by the 'newDeltaWeightTermII_alphaProduct' term.
                    
                    Matrix newDeltaWeightsTermII_alphaProduct_wrtNeuronE = oldDeltaWeightsFromEtoBandD.getProduct ( alpha ); //apha*oldDeltas
                    
                    Matrix newDeltaWeights_WrtNeuronEandD = newDeltaWeightsTermI_etaProduct_wrtNeuronE.getSum ( newDeltaWeightsTermII_alphaProduct_wrtNeuronE );
                    /////////////////////////
                    //update weights
                    LayerI_II_Weights_FromNeuronE.setMatrix ( LayerI_II_Weights_FromNeuronE.getSum ( newDeltaWeights_WrtNeuronEandD ) );
                    LayerI_II_DeltaWeights_FromNeuronE.setMatrix ( newDeltaWeights_WrtNeuronEandD );
                    
                    
                    
                //Layer 3 ops
                Matrix oldDeltaWeightsToNeuronG = LayerII_III_DeltaWeights_ToNeuronG; //old delta weights to neuron g
                Matrix oldWeightsToNeuronG = LayerII_III_Weights_ToNeuronG; //old weights to neuron g
                
                //new delta weight --> ((eta*thisGradient*priorOutcome) + (alpha*oldDelta))
                Matrix newDeltaWeightsToNeuronG = LayerII_Outcomes.getProduct ( eta * LayerIII_Gradients.getMatrix ( ) [ 0 ] [ 0 ] ).getSum ( oldDeltaWeightsToNeuronG.getProduct ( alpha ) );
                Matrix newWeightsToNeuronG = oldWeightsToNeuronG.getSum ( newDeltaWeightsToNeuronG );
                
                //update weights and delta weights
                    LayerII_III_Weights_ToNeuronG.setMatrix ( newWeightsToNeuronG );
                    LayerII_III_DeltaWeights_ToNeuronG.setMatrix ( newDeltaWeightsToNeuronG );
    }
    
    
    public static void renderMenu ( ) 
    {
        System.out.println ( "-------------------------------" );
        System.out.println ( "Test the neural network (aka supply unsupervised input) on (" + problemSpace.size ( ) + ") training samples." );
        System.out.println ( "1. Extract guess for Input = (1,1) " );
        System.out.println ( "2. Extract guess for Input = (1,0) " );
        System.out.println ( "3. Extract guess for Input = (0,1) " );
        System.out.println ( "4. Extract guess for Input = (0,0) " );
        System.out.println ( "5. Exit" );
        Scanner scanner = new Scanner ( System.in );
        int option = Integer.parseInt ( scanner.nextLine ( ) );
    
        switch ( option )
        {
            case 1:
            {
                Matrix inputValues = new Matrix ( 2, 1 );
                
                inputValues.setColumnMatrix ( new double [ ] { 1, 1 } );
                
                forwardPass ( inputValues );
                
                System.out.println ( "\n\nGuess : " + LayerIII_Outcomes.getMatrix ( ) [ 0 ] [ 0 ] + " (Correct Value is 0 for [1,1]) \n Press return to continue\n\n" );
                scanner.nextLine ( );
                System.out.println ( "\f" );
                renderMenu ( );
            }
            break;
           
            case 2:
            {
                Matrix inputValues = new Matrix ( 2, 1 );
                
                inputValues.setColumnMatrix ( new double [ ] { 1, 0 } );
                
                forwardPass ( inputValues );
                
                System.out.println ( "\n\nGuess : " + LayerIII_Outcomes.getMatrix ( ) [ 0 ] [ 0 ] + " (Correct Value is 1 for [1,0]) \n Press return to continue\n\n" );
                scanner.nextLine ( );
                System.out.println ( "\f" );
                renderMenu ( );
            }
            break;
            
            case 3:
            {
                Matrix inputValues = new Matrix ( 2, 1 );
                
                inputValues.setColumnMatrix ( new double [ ] { 0, 1 } );
                
                forwardPass ( inputValues );
                
                System.out.println ( "\n\nGuess : " + LayerIII_Outcomes.getMatrix ( ) [ 0 ] [ 0 ] + " (Correct Value is 1 for [0,1]) \n Press return to continue\n\n" );
                scanner.nextLine ( );
                System.out.println ( "\f" );
                renderMenu ( );
            }
            break;
            
            case 4:
            {
                Matrix inputValues = new Matrix ( 2, 1 );
                
                inputValues.setColumnMatrix ( new double [ ] { 0, 0 } );
                
                forwardPass ( inputValues );
                
                System.out.println ( "\n\nGuess : " + LayerIII_Outcomes.getMatrix ( ) [ 0 ] [ 0 ] + " (Correct Value is 0 for [0,0]) \n Press return to continue\n\n" );
                scanner.nextLine ( );
                System.out.println ( "\f" );
                renderMenu ( );
            }
            break;    
            
            
            case 5:
            {
                System.exit ( 0 );
            }
            break;
        } 
    }
}
