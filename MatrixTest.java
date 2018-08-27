//author: Jordan Micah Bennett


public class MatrixTest
{
    public static void main ( String [ ] arguments )
    { 
        Matrix mA = new Matrix ( 2, 2 );
        Matrix mB = new Matrix ( 2, 2 );
        
        
       
        mA.setMatrixByString ( "6,3x8,2" );
        mB.setMatrixByString ( "1,5x4,7" );

        System.out.println ( "Input String: 6,3x8,2" );
        System.out.println ( mA.describe ( ) );
        
        System.out.println ( "Input String: 1,5x4,7" );
        System.out.println ( mB.describe ( ) );

        System.out.println ( "\n\nExpected Multiplication Outcome:" );
        System.out.println ( "18,51" );
        System.out.println ( "16,54" );
        System.out.println ( "\nActual Multiplication Outcome:" );
        mA.getProduct ( mB );
        System.out.println ( mA.getProduct ( mB ) );

        System.out.println ( "\n\nExpected Addition Outcome:" );
        System.out.println ( "7,8" );
        System.out.println ( "12,9" );    
        System.out.println ( "\nActual Addition Outcome:" );
        mA.getSum ( mB );
        System.out.println ( mA.getSum ( mB ) );

        System.out.println ( "\n\nExpected Subtraction Outcome:" );
        System.out.println ( "5,-2" );
        System.out.println ( "4,-5" );
        System.out.println ( "\nActual Subtraction Outcome:" );
        mA.getDifference ( mB );
        System.out.println ( mA.getDifference ( mB ) );
        
        System.out.println ( "\n\nExpected Set Outcome:" );
        System.out.println ( mB.describe ( ) );
        System.out.println ( "\nActual Set Outcome:" );
        mA.setMatrix ( mB );
        System.out.println ( mA.describe ( ) );
        
        System.out.println ( "\n\nEmpty Matrix:" );
        Matrix mC = new Matrix ( 2, 1 );
        System.out.println ( mC.describe ( ) );
         
        System.out.println ( "\n\nActual Set Column Matrix Outcome:" );
        mC.setColumnMatrix ( new double [ ] { 4, 2 } );
        System.out.println ( mC.describe ( ) );
        
        System.out.println ( "\n\nActual self getSum outcome" );
        mC.getSum ( );
        System.out.println ( mC.describe ( ) ); 
        
        
        mA = new Matrix ( 2, 1 );
        mB = new Matrix ( 2, 1 );
        mA.setColumnMatrix ( new double [ ] { 1, 2 } );
        mB.setColumnMatrix ( new double [ ] { 4, 5 } );     
        System.out.println ( "\n\nOrder of methods is important:" );

        System.out.println ( "mA.getProduct ( mB.getPrimeActivation ( ) ) --> " );
        System.out.println ( mA.getProduct ( mB.getPrimeActivation ( ) ) ); //get product of a) mA and b) mB, then get c) prime activation of that product
        System.out.println ( "mB.getPrimeActivation ( ).getProduct ( mA ) --> " );
        System.out.println ( mB.getPrimeActivation ( ).getProduct ( mA ) ); //get product of a) prime activation on mB and b) mA
 
       
        Matrix mY = new Matrix ( 3, 1 );
        Matrix mZ = new Matrix ( 3, 1 );

        mY.setColumnMatrix ( new double [ ] { 1, 2, 3 } );
        mZ.setColumnMatrix ( new double [ ] { 4, 5, 6 } );
        System.out.println ( "n\nmY" + mY );
        
        System.out.println ( "mZ" + mZ + "\n\n" );
        
        System.out.println ( "mY.getAccumulatedDotProductWithoutSingularSum (mZ) --> " + mY.getAccumulatedDotProductWithoutSingularSum ( mZ ) );
    
        
        
    }
}