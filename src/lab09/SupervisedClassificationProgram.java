package lab09;

import java.io.*;
import java.util.*;

import weka.classifiers.*;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;

public class SupervisedClassificationProgram
{
	
    private static final String PATH_TO_FILE = "D:\\Programming\\Cygwin64\\home\\Lei Zhao\\"
    									+ "pivoted_genusLogNormalWithMetadata.arff";
 	
    public static void main(String[] args) throws Exception
    {
    	int numIterations = 100;
		int numThreads = 4;
		long startTime = System.currentTimeMillis();
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
