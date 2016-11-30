
package lab09;

import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.DoubleAdder;

import weka.classifiers.*;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;

public class SupervisedClassificationProgram
{
	
    private String file = "D:\\Programming\\Cygwin64\\home\\Lei Zhao\\"
    									+ "pivoted_genusLogNormalWithMetadata.arff";
    private static final int NUM_0F_ITERATIONS = 25;
    private static final int NUM_OF_THREADS = 4;
    private static final int SINGLE_T_TIMES = 100;
    private static final Random random = new Random(0);
    private static final DoubleAdder total = new DoubleAdder();
    

    public static void main(String[] args) throws Exception
    {
    	
		long startT = System.currentTimeMillis();
		System.out.println("Single Threading:");
		SupervisedClassificationProgram scp = new SupervisedClassificationProgram();
		System.out.println(scp.run(SINGLE_T_TIMES));
		
		long finishT = System.currentTimeMillis();
		System.out.println("Single Thread Running Time: " + (finishT - startT) / 1000f + "Seconds");
		
		System.out.println("Multi Threading:");
        System.out.println("Number of Processors: " + Runtime.getRuntime().availableProcessors());
    	
        startT = System.currentTimeMillis();
       
        for(int i=0; i<NUM_OF_THREADS; i++)
    	{
        	SupervisedClassificationProgram mu = new SupervisedClassificationProgram();
    		mu.start();
    	}
        
        finishT = System.currentTimeMillis();
		System.out.println("Multi Thread Running Time: " + (finishT - startT) / 1000f + "Seconds");
        
    }
    
    private Double run(int numPermutations) throws Exception
	{	
    		File aFile = new File(file);
    		List<Double> percentCorrect = getPercentCorrectForOneFile(aFile, numPermutations, random);
		
    		for(Iterator<Double> itr = percentCorrect.iterator();itr.hasNext();)
    		{
    			if(itr.hasNext())
    			{
    				System.out.println(itr.next());
    			}
    		}
    		return new Double (percentCorrect.size());
	}
    
    private void start()
    {
    	Runnable r = new Runnable()
    	{
    		@Override
    		public void run()
    		{
        	    try
				{
        	    	total.add(SupervisedClassificationProgram.this.run(NUM_0F_ITERATIONS));
				} 	catch (Exception e)
				
        	    {
					e.printStackTrace();
				}
    			
    		}
    	};
    	new Thread(r).start();
    }
    
    public static List<Double> getPercentCorrectForOneFile( File inFile, int numPermutations, Random random ) 
			throws Exception
    {
    	List<Double> percentCorrect = new ArrayList<Double>();
	
    	for( int x=0; x< numPermutations; x++)
    	{
    		Instances data = DataSource.read(inFile.getAbsolutePath());
    		data.setClassIndex(data.numAttributes() -1);
    		Evaluation ev = new Evaluation(data);
    		AbstractClassifier rf = new RandomForest();
		
    		//rf.buildClassifier(data);
    		ev.crossValidateModel(rf, data, 10, random);
    		//System.out.println(ev.toSummaryString("\nResults\n\n", false));
    		//System.out.println(x + " " + ev.areaUnderROC(0) + " " + ev.pctCorrect());
    		percentCorrect.add(ev.pctCorrect());
    	}
	
    	return percentCorrect;	
    }

}