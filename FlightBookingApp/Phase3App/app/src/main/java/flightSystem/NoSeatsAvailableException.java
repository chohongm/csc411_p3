package flightSystem;

/**
 * An exception to be thrown if there are no seats available on an itinerary.
 */
public class NoSeatsAvailableException extends Exception {

    public NoSeatsAvailableException() {

    }

    public NoSeatsAvailableException(String message){
        super(message);
    }
}
