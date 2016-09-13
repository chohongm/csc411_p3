package flightSystem;

/**
 * An exception to be thrown when a client tries to book an itinerary twice.
 */
public class ClientAlreadyBookedException extends Exception {

    public ClientAlreadyBookedException() {

    }

    public ClientAlreadyBookedException(String message) {
        super(message);
    }
}
