package edu.toronto.csc301;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.Timeout;

import static edu.toronto.csc301.util.Asserts.*;


public class SetupTest {

	// Define a rule - Each test method has (at most) 2 seconds to run.
	@Rule  
    public Timeout globalTimeout = Timeout.seconds(2);
	
	
	public static final String  BASIC_ROBOT = "edu.toronto.csc301.BasicRobot";
	
	
	@Test
	public void checkExistence_BasicRobot() {
		assertClassExists(BASIC_ROBOT);
	}
	
	@Test
	public void checkInterface_BasicRobot() throws ClassNotFoundException {
		assertClassImplementsInterface(BASIC_ROBOT, IBasicRobot.class);
	}
	
	@Test
	public void checkConstructor_BasicRobot() throws ClassNotFoundException {
		assertClassHasConstructor(BASIC_ROBOT, Double.TYPE, Double.TYPE, Integer.TYPE);
	}
	

}
