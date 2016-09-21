package edu.toronto.csc301;


public interface IBasicRobot {
	public double getXCoordinate();
	public double getYCoordinate();
	public int getRotation();

	public void rotateRight(int degrees);
	public void rotateLeft(int degrees);
	public void moveForward(int millimeters);
}
