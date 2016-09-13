package flightSystem;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * An abstract class that represents someone who uses the flight system. Stores
 * personal user information.
 *
 */
public abstract class User implements Serializable {
	
	//The last name of the user.
	protected String lastName;
	
	//The first name of the user.
	protected String firstName;
	
	//The email of the user.
	protected String email;
	
	//The address of the user.
	protected String address;

	//All the itineraries this client is booked on
	protected ArrayList<Itinerary> bookedItineraries;
	
	/**
	 * Create a user with the given personal information.
	 * @param lastName The last name of the user.
	 * @param firstName The first name of the user.
	 * @param email The email of the user.
	 * @param address The address of the user.
	 */
	public User(String lastName, String firstName, String email, 
			String address) {
		
		this.lastName = lastName;
		this.firstName = firstName;
		this.email = email;
		this.address = address;
		this.bookedItineraries = new ArrayList<Itinerary>();//load it first
	}

	/**
     * Gets the last name of this user.
	 * @return The last name of this user.
	 */
	public String getLastName() {
		return lastName;
	}

	/**
     * Sets the last name for this user
	 * @param lastName The last name to set for this user.
	 */
	public void setLastName(String lastName) {
		this.lastName = lastName;
	}

	/**
     * Gets the first name for this user.
	 * @return The first name of this user.
	 */
	public String getFirstName() {
		return firstName;
	}

	/**
     * Sets the first name for this user.
	 * @param firstName The first name to set for this user.
	 */
	public void setFirstName(String firstName) {
		this.firstName = firstName;
	}

	/**
     * Gets the email of this user.
	 * @return The email of this user.
	 */
	public String getEmail() {
		return email;
	}

	/**
     * Sets the email of this user.
	 * @param email The email to set for this user.
	 */
	public void setEmail(String email) {
		this.email = email;
	}

	/**
     * Gets the address of this user.
	 * @return The address of this user.
	 */
	public String getAddress() {
		return address;
	}

	/**
     * Sets the address of this user.
	 * @param address The address to set for this user.
	 */
	public void setAddress(String address) {
		this.address = address;
	}
	
	/**
     * Gives a string representation of this user's information separated
     * by commas.
	 * @return A string representation containing this user's personal 
	 * information.
	 */
	public String toString() {
		return lastName + "," + firstName + "," + email + "," + address;
	}

    /**
     * Checks for equality between this User and another object.
     * @param other The object to compare.
     * @return True if and only if other is a User and the email of other
     * equals the email of this user.
     */
	public boolean equals(Object other){
		return (other instanceof User)
                && ((User) other).getEmail().equals(this.email);
	}
}
