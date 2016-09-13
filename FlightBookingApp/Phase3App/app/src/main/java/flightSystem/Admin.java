package flightSystem;

import java.io.Serializable;

/**
 * Represents an administrator for this application.
 * 
 */
public class Admin extends User implements Serializable{

	/**
	 * Create an administrator with the given personal information.
	 * @param lastName The last name of the user.
	 * @param firstName The first name of the user.
	 * @param email The email of the user.
	 * @param address The address of the user.
	 */
	public Admin(String lastName, String firstName, String email, 
			String address) {
		
		super(lastName, firstName, email, address);
	}

}
