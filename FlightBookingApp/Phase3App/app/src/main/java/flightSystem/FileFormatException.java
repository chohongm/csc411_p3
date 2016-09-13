package flightSystem;

/**
 * An exception to be thrown if there is not enough or too much information 
 * in a upload file.
 * 
 */
public class FileFormatException extends Exception {
	
	public FileFormatException() {

	}

	public FileFormatException(String message) {
		super(message);
	}
}
