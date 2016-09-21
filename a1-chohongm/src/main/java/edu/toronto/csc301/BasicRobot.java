package edu.toronto.csc301;

public class BasicRobot implements IBasicRobot {
	private double x;
	private double y;
	private int z;
	
	public BasicRobot(double x, double y, int z) {
		this.x = x;
		this.y = y;
		if (z >= 360) {
			throw new IllegalArgumentException();
		} else if (z < 0) {
			throw new IllegalArgumentException();
		}
		this.z = (z % 360);
	}

	@Override
	public double getXCoordinate() {
		// TODO Auto-generated method stub
		return x;
	}

	@Override
	public double getYCoordinate() {
		// TODO Auto-generated method stub
		return y;
	}

	@Override
	public int getRotation() {
		// TODO Auto-generated method stub
		return z;
	}

	@Override
	public void rotateRight(int degrees) {
		// TODO Auto-generated method stub
		this.z += degrees;
		if (this.z < 0){
			this.z += 360;
		} else {
			this.z = this.z % 360;

		}
		//System.out.println(this.z);
	}

	@Override
	public void rotateLeft(int degrees) {
		// TODO Auto-generated method stub
		this.z -= degrees;
		this.z = this.z % 360;

		if (this.z < 0){
			this.z += 360;
		}
	}

	@Override
	public void moveForward(int millimeters) {
		// TODO Auto-generated method stub
		y += Math.cos(Math.toRadians(z)) * millimeters;
		x += Math.sin(Math.toRadians(z)) * millimeters;
	}
}