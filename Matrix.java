//Author: Jordan Micah Bennett
//Purpose: Matrix Representation written from scratch for Artificial Neural Network usage. Includes addition, multiplication, and subtraction operations (including scalar operations on Matrix)

import java.util.Random;

public class Matrix
{
    private double [ ] [ ] _Matrix = null;
    private int rowCardinality, columnCardinality;
   
    public Matrix ( int rowCardinality, int columnCardinality )
    {
        this.rowCardinality = rowCardinality;
        this.columnCardinality = columnCardinality;
        
        _Matrix = new double [ this.rowCardinality ] [ this.columnCardinality ];
    }
    
    public double [ ] [ ] getMatrix ( ) 
    {
        return _Matrix;
    }
    
    public Matrix getMatrix ( double [ ] [ ] _Matrix ) 
    {
        //Where param 0: total number of rows is esentially the same as matrix size
        //Where param 1: //size of any row, (aka column) given that each row has equal numbers of elements
        Matrix returnValue = new Matrix ( _Matrix.length, _Matrix [ 0 ].length );

        returnValue.setMatrix ( _Matrix );
        
        return returnValue;
    }
    /*    
        Notes:
        
        a) 2 loops allow an algorithm to explore some structure, on a cell by cell basis, while visiting each new cell once.
        Eg: for ( int i = 0; i < someLength; i ++ ) for ( int j = 0; j < someLength; j ++ ) 
        
        b) 3 loops allow an algorithm to explore some structure, on a cell by cell basis, while visiting each new cell twice. 
        ... This is relevant because each cell or multiplication entry is incomplete in two loop system, 
        ... as two loops permit consumption of only one element from a row and column.
        ... Matrix multiplication requires consumption of an entire row, and an entire column, per calculated entry, so two loops would be insufficient.
          
        c) Each inner loop consumes all elements per limit or increments until limit, then permits increment index of outer loop to increase.
        
        d) In the 3 loop system, all indices need to correlate with the reality that such a system permits cell visiting twice.
        
    */
    public Matrix getProduct ( Matrix _Factor ) //where factor is another matrix, which will multiply by each matrix entry of this class
    {
        double [ ] [ ] returnValue = new double [ this.getMatrix ( ).length ] [ _Factor.getMatrix ( ) [ 0 ].length ]; //declare product matrix (which is the returnValue) with row size of this matrix, and column size of _Factor

        int totalCardinality_thisRow = this.getMatrix ( ).length; //total number of rows or "totalCardinality_thisRow" is esentially the same as matrix size
        int totalCardinality_thisColumn = this.getMatrix ( ) [ 0 ].length; //size of any row, (aka column) given that each row has equal numbers of elements
        int totalCardinality_factorColumn = _Factor.getMatrix ( ) [ 0 ].length; //size of any row, (aka column) given that each row has equal numbers of elements
        
        //define product matrix
        for ( int rMI = 0; rMI < totalCardinality_thisRow; rMI ++ )
            for ( int fCI = 0; fCI < totalCardinality_factorColumn; fCI ++ )
                for ( int mCI = 0; mCI < totalCardinality_thisColumn; mCI ++ )
                    returnValue [ rMI ] [ fCI ] += this.getMatrix ( ) [ rMI ] [ mCI ] * _Factor.getMatrix ( ) [ mCI ] [ fCI ];                       
                
        //describe matrix
        //System.out.println ( describe ( returnValue ) );
        
        return getMatrix ( returnValue );
    }
    
    
    public Matrix getProduct ( double _Factor ) //where factor is double, which will multiply by each matrix entry of this class
    {
        double [ ] [ ] returnValue = new double [ this.getMatrix ( ).length ] [ this.getMatrix ( ) [ 0 ].length ]; //declare product matrix (which is the returnValue) with row size of this matrix, and column size of _Factor

        //define product matrix
        for ( int rMI = 0; rMI < this.getMatrix ( ).length; rMI ++ )
            for ( int mCI = 0; mCI < this.getMatrix ( ) [ 0 ].length; mCI ++ )
                returnValue [ rMI ] [ mCI ] += this.getMatrix ( ) [ rMI ] [ mCI ] * _Factor;                    
                
        //describe matrix
        //System.out.println ( describe ( returnValue ) );
        
        return getMatrix ( returnValue );
    }   
    
    //this method returns an accumulation of dot products, in a way that:
    //1) Does not return a 1x1 matrix/scalar
    //2) Does the above while adding prior product to each subsequent product entry generated
    public Matrix getAccumulatedDotProductWithoutSingularSum ( Matrix _Factor ) //where factor is another matrix, which will multiply by each matrix entry of this class
    {
        double [ ] [ ] returnValue = new double [ this.getMatrix ( ).length ] [ _Factor.getMatrix ( ) [ 0 ].length ]; //declare product matrix (which is the returnValue) with row size of this matrix, and column size of _Factor
        
        double priorSigma = 0, sigma = 0;
        
        //define product matrix
        for ( int rI = 0; rI < returnValue.length; rI ++ ) //where rI = row iterator
            for ( int cI = 0; cI < returnValue [ 0 ].length; cI ++ ) //where cI = column iterator
            {
                priorSigma = ( rI - 1 > -1 && cI - 1 > -1 ) ? this.getMatrix ( ) [ rI - 1 > -1 ? rI - 1 : 0 ] [ cI - 1 > -1 ? cI - 1 : 0 ] * _Factor.getMatrix ( ) [ rI - 1 > -1 ? rI - 1 : 0 ] [ cI - 1 > -1 ? cI - 1 : 0 ] : 0;
                sigma += this.getMatrix ( ) [ rI ] [ cI ] * _Factor.getMatrix ( ) [ rI ] [ cI ];

                returnValue [ rI ] [ cI ] = sigma + priorSigma; 
            }
                
        //describe matrix
        //System.out.println ( describe ( returnValue ) );
        
        return getMatrix ( returnValue );
    }
 
    
    public Matrix getSum ( Matrix _Factor ) //where factor is another matrix, which will be added to each matrix entry of this class
    {
        double [ ] [ ] returnValue = new double [ this.getMatrix ( ).length ] [ _Factor.getMatrix ( ) [ 0 ].length ]; //declare product matrix (which is the returnValue) with row size of this matrix, and column size of _Factor

        //define sum matrix (by Matrix _Factor)
        for ( int rI = 0; rI < returnValue.length; rI ++ ) //where rI = row iterator
            for ( int cI = 0; cI < returnValue [ 0 ].length; cI ++ ) //where cI = column iterator
                returnValue [ rI ] [ cI ] = this.getMatrix ( ) [ rI ] [ cI ] + _Factor.getMatrix ( ) [ rI ] [ cI ];  
        
        //describe matrix
        //System.out.println ( describe ( returnValue ) );
        
        return getMatrix ( returnValue );
    }
    
    public Matrix getSum ( double _Factor ) //where factor is as a scalar, which will added to each matrix entry
    {
        double [ ] [ ] returnValue = new double [ this.getMatrix ( ).length ] [ this.getMatrix ( ) [ 0 ].length ]; //declare product matrix (which is the returnValue) with row size of this matrix, and column size of _Factor

        //define sum matrix (by Matrix _Factor)
        for ( int rI = 0; rI < returnValue.length; rI ++ ) //where rI = row iterator
            for ( int cI = 0; cI < returnValue [ 0 ].length; cI ++ ) //where cI = column iterator
                returnValue [ rI ] [ cI ] += this.getMatrix ( ) [ rI ] [ cI ] + _Factor;  
        
        //describe matrix
        //System.out.println ( describe ( returnValue ) );
        
        return getMatrix ( returnValue );
    }
    
    public double getSum ( ) //returns sum of the current entries in matrix (a self sum). 
    {
        double returnValue = 0.0;
        
        //define sum matrix (by Matrix _Factor)
        for ( int rI = 0; rI < this.getMatrix ( ).length; rI ++ ) //where rI = row iterator
            for ( int cI = 0; cI < this.getMatrix ( ) [ 0 ].length; cI ++ ) //where cI = column iterator
                returnValue += this.getMatrix ( ) [ rI ] [ cI ];
        
        //describe matrix
        //System.out.println ( "_matrix_sum --> " + returnValue );
        
        return returnValue;
    }
    
    public Matrix getDifference ( Matrix _Factor ) //where factor is another matrix, which will subtract from each matrix entry in this class
    {
        double [ ] [ ] returnValue = new double [ this.getMatrix ( ).length ] [ _Factor.getMatrix ( ) [ 0 ].length ]; //declare product matrix (which is the returnValue) with row size of this matrix, and column size of _Factor

        //define sum matrix (by Matrix _Factor)
        for ( int rI = 0; rI < returnValue.length; rI ++ ) //where rI = row iterator
            for ( int cI = 0; cI < returnValue [ 0 ].length; cI ++ ) //where cI = column iterator
                returnValue [ rI ] [ cI ] = this.getMatrix ( ) [ rI ] [ cI ] - _Factor.getMatrix ( ) [ rI ] [ cI ];  
        
        //describe matrix
        //System.out.println ( describe ( returnValue ) );
        
        return getMatrix ( returnValue );
    }
    
    public Matrix getDifference ( double _Factor ) //where factor is another matrix, which will subtract from each matrix entry in this class
    {
        double [ ] [ ] returnValue = new double [ this.getMatrix ( ).length ] [ this.getMatrix ( ) [ 0 ].length ]; //declare product matrix (which is the returnValue) with row size of this matrix, and column size of _Factor

        //define sum matrix (by Matrix _Factor)
        for ( int rI = 0; rI < returnValue.length; rI ++ ) //where rI = row iterator
            for ( int cI = 0; cI < returnValue [ 0 ].length; cI ++ ) //where cI = column iterator
                returnValue [ rI ] [ cI ] = this.getMatrix ( ) [ rI ] [ cI ] - _Factor;  
        
        //describe matrix
        //System.out.println ( describe ( returnValue ) );
        
        return getMatrix ( returnValue );
    }
    
    public String describe ( double [ ] [ ] _MatrixValue )
    {
        String returnValue = _MatrixValue.length + "x" + _MatrixValue [ 0 ].length + " matrix : \n";
        
        int newLineCounter = 1;
        
        for ( int rI = 0; rI < _MatrixValue.length /* row cardinality */; rI ++ ) 
            for ( int cI = 0; cI < _MatrixValue [ 0 ].length /* column cardinality */; cI ++ )
            {
                returnValue += _MatrixValue [ rI ] [ cI ] + ( cI > -1 && cI < _MatrixValue [ 0 ].length - 1 ? "," : "" ) + ( newLineCounter % 2 == 0 ? "\n" : " " );
                newLineCounter ++;
            }
            
        return returnValue;
    }
    
    public String describe ( )
    {
         return describe ( this.getMatrix ( ) );
    }
    
    public String toString ( )
    {
        return describe ( this.getMatrix ( ) );
    }
    
    public void setMatrixByString ( String description ) //eg nxn input = "1,2x3,4"
    {
        for ( int rI = 0; rI < rowCardinality; rI ++ )
            for ( int cI = 0; cI < columnCardinality; cI ++ )
                _Matrix [ rI ] [ cI ] = Double.parseDouble ( description.split ( "x" ) [ rI ].split ( "," ) [ cI ] ); //rows are split by 'x', while columns split by ','
    }
    
    //Note: Column matrices are ones with only one column, and one or more rows
    public void setColumnMatrix ( double [ ] description ) //eg for nx1 matrix, input = [1,2,3], where n = each row entry.
    {
        for ( int rI = 0; rI < description.length; rI ++ )
            for ( int cI = 0; cI < columnCardinality; cI ++ )
                _Matrix [ rI ] [ cI ] = description [ rI ]; 
    } 

    
    public void setMatrix  ( Matrix _Factor ) //where factor is another matrix, which will establish this matrix in _Factor terms
    {
        //define sum matrix (by Matrix _Factor)
        for ( int rI = 0; rI < _Factor.getMatrix ( ).length; rI ++ ) //where rI = row iterator //Bound limit to target value's length
            for ( int cI = 0; cI < this.getMatrix ( ) [ 0 ].length; cI ++ ) //where cI = column iterator
                this.getMatrix ( ) [ rI ] [ cI ] = _Factor.getMatrix ( ) [ rI ] [ cI ];  
        
        //describe matrix
        //System.out.println ( describe ( this.getMatrix ( ) ) );
    }
    
    public void setMatrix  ( double _Factor ) //where factor is another matrix, which will establish this matrix in _Factor terms
    {
        //define sum matrix (by Matrix _Factor)
        for ( int rI = 0; rI < this.getMatrix ( ).length; rI ++ ) //where rI = row iterator
            for ( int cI = 0; cI < this.getMatrix ( ) [ 0 ].length; cI ++ ) //where cI = column iterator
                this.getMatrix ( ) [ rI ] [ cI ] = _Factor;  
        
        //describe matrix
        //System.out.println ( describe ( this.getMatrix ( ) ) );
    }    
    
    public void setMatrix  ( double [ ] [ ] _Factor ) //where factor is another matrix, which will establish this matrix in _Factor terms
    {
        //define sum matrix (by double [ ] [ ] _Factor)
        for ( int rI = 0; rI < _Factor.length; rI ++ ) //where rI = row iterator
            for ( int cI = 0; cI < this.getMatrix ( ) [ 0 ].length; cI ++ ) //where cI = column iterator
                this.getMatrix ( ) [ rI ] [ cI ] = _Factor [ rI ] [ cI ];  
        
    }
    
    public void randomize  ( ) 
    {
        //define sum matrix (by Matrix _Factor)
        for ( int rI = 0; rI < this.getMatrix ( ).length; rI ++ ) //where rI = row iterator
            for ( int cI = 0; cI < this.getMatrix ( ) [ 0 ].length; cI ++ ) //where cI = column iterator
                this.getMatrix ( ) [ rI ] [ cI ] = new Random ( ).nextDouble ( );  
        
        //describe matrix
        //System.out.println ( describe ( this.getMatrix ( ) ) );
    }
    
    //derivative
    public Matrix getActivation ( )
    {
        double [ ] [ ] returnValue = new double [ this.getMatrix ( ).length ] [ this.getMatrix ( ) [ 0 ].length ];
        
        //define sum matrix (by Matrix _Factor)
        for ( int rI = 0; rI < this.getMatrix ( ).length; rI ++ ) //where rI = row iterator
            for ( int cI = 0; cI < this.getMatrix ( ) [ 0 ].length; cI ++ ) //where cI = column iterator
                returnValue [ rI ] [ cI ] = Math.tanh ( this.getMatrix ( ) [ rI ] [ cI ] );
        
        return this.getMatrix ( returnValue );
    }
    public double getActivation ( double value )
    {
        return Math.tanh ( value );
    }
    
    //second derivative
    public Matrix getPrimeActivation ( )
    {
        double [ ] [ ] returnValue = new double [ this.getMatrix ( ).length ] [ this.getMatrix ( ) [ 0 ].length ];
        
        //define sum matrix (by Matrix _Factor)
        for ( int rI = 0; rI < this.getMatrix ( ).length; rI ++ ) //where rI = row iterator
            for ( int cI = 0; cI < this.getMatrix ( ) [ 0 ].length; cI ++ ) //where cI = column iterator
                returnValue [ rI ] [ cI ] = 1 - Math.pow ( Math.tanh ( this.getMatrix ( ) [ rI ] [ cI ] ), 2 );
                
        return this.getMatrix ( returnValue );
    }
    public double getPrimeActivation ( double value )
    {
        return 1 - Math.pow ( Math.tanh ( value ), 2 );
    }
}