package edu.toronto.csc301;

import static edu.toronto.csc301.util.Helpers.newInstance;
import static org.junit.Assert.assertEquals;

import java.util.Random;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.Timeout;

public class BasicRobotTest {

	@Rule
    public Timeout globalTimeout = Timeout.seconds(2);	
	private Random random = new Random();
	
	
	// =============================== Tests ==================================
	
	
	
	@Test
	public void createRobotAndCheckXCoordinate() throws Exception {
		IBasicRobot robot = createBasicRobot(1.5, -3.1, 60);
		assertEquals(1.5, robot.getXCoordinate(), 0.01);
	}
	
	@Test
	public void createRobotAndCheckYCoordinate() throws Exception {
		IBasicRobot robot = createBasicRobot(1.5, -3.1, 60);
		assertEquals(-3.1, robot.getYCoordinate(), 0.01);
	}
	
	@Test
	public void createRobotAndCheckRotation() throws Exception {
		IBasicRobot robot = createBasicRobot(1.5, -3.1, 60);
		assertEquals(60, robot.getRotation());
	}
	
	
	
	
	@Test
	public void createRobotWithRandomLocationAndRotationAndCheckXCoordinate() throws Exception {
		double x     = (random.nextDouble() - 0.5) * 100;
		double y     = (random.nextDouble() - 0.5) * 100;
		int rotation = random.nextInt(360);
		IBasicRobot robot = createBasicRobot(x, y, rotation);
		assertEquals(x, robot.getXCoordinate(), 0.01);
	}
	
	@Test
	public void createRobotWithRandomLocationAndRotationAndCheckYCoordinate() throws Exception {
		double x     = (random.nextDouble() - 0.5) * 100;
		double y     = (random.nextDouble() - 0.5) * 100;
		int rotation = random.nextInt(360);
		IBasicRobot robot = createBasicRobot(x, y, rotation);
		assertEquals(y, robot.getYCoordinate(), 0.01);
	}
	
	@Test
	public void createRobotWithRandomLocationAndRotationAndCheckRotation() throws Exception {
		double x     = (random.nextDouble() - 0.5) * 100;
		double y     = (random.nextDouble() - 0.5) * 100;
		int rotation = random.nextInt(360);
		IBasicRobot robot = createBasicRobot(x, y, rotation);
		assertEquals(rotation, robot.getRotation());
	}
	
	
	// ------------------------------------------------------------------------
	
	
	
	@Test
	public void createRobotWithRotationZero() throws Exception {
		createBasicRobot(0, 0, 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void cannotCreateRobotWithNegativeRotation() throws Exception {
		createBasicRobot(0, 0, -1);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void cannotCreateRobotWithRandomNegativeRotation() throws Exception {
		int rotation = -1 * (Math.abs(random.nextInt()) + 1);
		createBasicRobot(0,0, rotation);
	}
	
	
	@Test
	public void createRobotWithRotation359() throws Exception {
		createBasicRobot(0, 0, 359);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void cannotCreateRobotWithRotationThatIsTooLarge() throws Exception {
		createBasicRobot(0,0, 360);
	}

	@Test(expected=IllegalArgumentException.class)
	public void cannotCreateRobotWithRandomRotationThatIsTooLarge() throws Exception {
		int rotation = 360 + Math.abs(random.nextInt());
		createBasicRobot(0,0, rotation);
	}
	
	
	

	// ------------------------------------------------------------------------


	@Test
	public void rotateRight_test1() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 0);
		robot.rotateRight(45);
		assertEquals(45, robot.getRotation());
	}

	@Test
	public void rotateRight_test2() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 0);
		robot.rotateRight(373);
		assertEquals(13, robot.getRotation());
	}

	@Test
	public void rotateRight_test3() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 0);
		robot.rotateRight(-59);
		assertEquals(301, robot.getRotation());
	}


	
	@Test
	public void rotateLeft_test1() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 0);
		robot.rotateLeft(45);
		assertEquals(315, robot.getRotation());
	}

	@Test
	public void rotateLeft_test2() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 0);
		robot.rotateLeft(373);
		assertEquals(347, robot.getRotation());
	}

	@Test
	public void rotateLeft_test3() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 0);
		robot.rotateLeft(-59);
		assertEquals(59, robot.getRotation());
	}


	// ------------------------------------------------------------------------


	@Test
	public void moveNorth() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 0);
		robot.moveForward(1000);
		assertRobotLocation(robot, 0, 1000);
	}

	@Test
	public void moveEast() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 90);
		robot.moveForward(1000);
		assertRobotLocation(robot, 1000, 0);
	}

	@Test
	public void moveSouth() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 180);
		robot.moveForward(1000);
		assertRobotLocation(robot, 0, -1000);
	}

	@Test
	public void moveWest() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 270);
		robot.moveForward(1000);
		assertRobotLocation(robot, -1000, 0);
	}


	// ------------------------------------------------------------------------


	@Test
	public void moveNorth_test2() throws Exception {
		IBasicRobot robot = createBasicRobot(24,56, 0);
		robot.moveForward(1000);
		assertRobotLocation(robot, 24, 1056);
	}

	@Test
	public void moveEast_test2() throws Exception {
		IBasicRobot robot = createBasicRobot(99,-32, 90);
		robot.moveForward(1000);
		assertRobotLocation(robot, 1099, -32);
	}

	@Test
	public void moveSouth_test2() throws Exception {
		IBasicRobot robot = createBasicRobot(7,200, 180);
		robot.moveForward(1000);
		assertRobotLocation(robot, 7, -800);
	}

	@Test
	public void moveWest_test2() throws Exception {
		IBasicRobot robot = createBasicRobot(1000,1, 270);
		robot.moveForward(1000);
		assertRobotLocation(robot, 0, 1);
	}


	// ------------------------------------------------------------------------


	@Test
	public void moveWhenRotationIs45Degrees() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 45);
		robot.moveForward(1000);
		assertRobotLocation(robot, 707.106, 707.106);
	}

	@Test
	public void moveWhenRotationIs60Degrees() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 60);
		robot.moveForward(1000);
		assertRobotLocation(robot, 866.025, 500);
	}

	@Test
	public void moveWhenRotationIs150Degrees() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 150);
		robot.moveForward(1000);
		assertRobotLocation(robot, 500, -866.025);
	}

	@Test
	public void moveWhenRotationIs230Degrees() throws Exception {
		IBasicRobot robot = createBasicRobot(0,0, 230);
		robot.moveForward(1000);
		assertRobotLocation(robot, -766.044, -642.787);
	}


	
	// ------------------------------------------------------------------------
	
	
	@Test
	public void rotateAndMove_test1() throws Exception{
		IBasicRobot robot = createBasicRobot(0,0, 0);
		robot.rotateRight(45);
		robot.moveForward(1000);
		assertRobotLocation(robot, 707.106, 707.106);
	}
	
	@Test
	public void rotateAndMove_test2() throws Exception{
		IBasicRobot robot = createBasicRobot(0,0,0);
		robot.rotateLeft(45);
		robot.moveForward(1000);
		assertRobotLocation(robot, -707.106, 707.106);
	}
	
	@Test
	public void rotateAndMove_test3() throws Exception{
		IBasicRobot robot = createBasicRobot(0,0, 60);
		robot.rotateRight(300);
		robot.rotateLeft(45);
		robot.moveForward(1000);
		assertRobotLocation(robot, -707.106, 707.106);
	}

	
	@Test
	public void rotateAndMove_test4() throws Exception{
		double x     = (random.nextDouble() - 0.5) * 100;
		double y     = (random.nextDouble() - 0.5) * 100;
		int rotation = random.nextInt(360);
		int step = 42;
		
		IBasicRobot robot = createBasicRobot(x, y, rotation);
		
		// Repeat the loop 4 times
		for (int i = 0; i < 4; i++) {
			robot.rotateRight(90);
			robot.moveForward(step);
		}
		
		// We expect the robot to end up where it started
		assertRobotLocation(robot, x, y);
	}
	
	
	@Test
	public void rotateAndMove_test5() throws Exception{
		double x     = (random.nextDouble() - 0.5) * 100;
		double y     = (random.nextDouble() - 0.5) * 100;
		int rotation = random.nextInt(360);
		int step = 42;
		
		IBasicRobot robot = createBasicRobot(x, y, rotation);
		
		// Repeat the loop 4 times
		for (int i = 0; i < 4; i++) {
			robot.rotateLeft(90);
			robot.moveForward(step);
		}
		
		// We expect the robot to end up where it started
		assertRobotLocation(robot, x, y);
	}
	
	
	
	// =============================== Helpers ================================

	
	/**
	 * Assert that the given <code>robot</code> is currently at the location
	 * specified by the given <code>x</code> and <code>y</code> coordinates.
	 */
	private void assertRobotLocation(IBasicRobot robot, double x, double y){
		assertEquals(x, robot.getXCoordinate(), 0.1);
		assertEquals(y, robot.getYCoordinate(), 0.1);
	}
	
	
	/**
	 * Helper function that creates (and returns) an instance of your
	 * IBasicRobot implementation.
	 * 
	 * The arguments passed to this method will be passed to the constructor
	 * of your IBasicRobot implementation.
	 */
	private IBasicRobot createBasicRobot(double x, double y, int rotation) 
															throws Exception{
		return newInstance(SetupTest.BASIC_ROBOT, 
				new Class<?>[]{Double.TYPE, Double.TYPE, Integer.TYPE}, 
				x, y, rotation);
	}
	
	// ========================================================================



}
