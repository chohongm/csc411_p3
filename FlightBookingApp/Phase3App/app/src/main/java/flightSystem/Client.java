package flightSystem;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Represents a client who uses this application to search for and book flights.
 * 
 */
public class Client extends User implements Serializable {
	
	//The credit card number of this client.
	private String creditCardNumber;
	
	//The expiry date of this client's credit card.
	private String expiryDate;

	//The itineraries that this client has booked.
	private List<Itinerary> bookedItineraries;

	/**
	 * Create a client with the given personal information.
	 * @param lastName The last name of this client.
	 * @param firstName The first name of this client.
	 * @param email The email address of this client.
	 * @param address The address of this client.
	 * @param creditCardNumber The credit card number of this client.
	 * @param expiryDate The expiry date of this client's credit card.
	 */
	public Client(String lastName, String firstName, String email, 
			String address, String creditCardNumber, String expiryDate) {
		
		super(lastName, firstName, email, address);
		this.creditCardNumber = creditCardNumber;
		this.expiryDate = expiryDate;
		this.bookedItineraries = new ArrayList<Itinerary>();
	}
	
	
	/**
	 * Gets the credit car number of this client.
	 * @return The credit card number of this client
	 */
	public String getCreditCardNumber() {
		return creditCardNumber;
	}


	/**
	 * Sets the credit card number of this client.
	 * @param creditCardNumber The credit card number to set for this client.
	 */
	public void setCreditCardNumber(String creditCardNumber) {
		this.creditCardNumber = creditCardNumber;
	}


	/**
     * Gets the expiry date of this client's credit card.
	 * @return The expiry date of this client's credit card.
	 */
	public String getExpiryDate() {
		return expiryDate;
	}


	/**
     * Sets the expiry date of this client's credit card.
	 * @param expiryDate The expiry date to set for this client's credit card.
	 */
	public void setExpiryDate(String expiryDate) {
		this.expiryDate = expiryDate;
	}

    /**
     * Books an itinerary for this client.
     * @param itinerary The itinerary to be booked.
     * @throws NoSeatsAvailableException If there are no seats available on
     * this itinerary.
     * @throws ClientAlreadyBookedException If the client was already booked
     * on this itinerary.
     */
	public void bookItinerary(Itinerary itinerary)
            throws NoSeatsAvailableException, ClientAlreadyBookedException {

        List<Flight> flightSequence = itinerary.getFlightSequence();

        //First, check all flights of the itinerary has a seat available
        // for booking.
        for (Flight f : flightSequence) {
            if (f.getNumAvailableSeats() == 0) {
                throw new NoSeatsAvailableException(
                        "The flight is booked at its full capacity.");

            //Check if this client is alread booked on a flight.
            } else if (f.getBookedClients().contains(this)) {
                throw new ClientAlreadyBookedException(
                        "The given client is already booked.");
            }
        }
        // seats are available for all flights of the itinerary, so book.
        for (Flight f : flightSequence) {
            f.bookClient(this);
        }

        //Add this itinerary to this client's list of booked itineraries.
        bookedItineraries.add(itinerary);
	}


    /**
     * Cancels the booking of the given itinerary for this client.
     * @param itinerary The itinerary to be canceled.
     */
	public void cancelItinerary(Itinerary itinerary) {
		List<Flight> flightSequence = (ArrayList<Flight>) itinerary.getFlightSequence();

        //Cancel each individual flight
		for (Flight f : flightSequence) {
            FlightSystem.getInstance().searchOneFlight(f).
                    cancelBookedClient(this);
		}

        //Remove from client's booked itineraries.
		bookedItineraries.remove(itinerary);
	}

	/**
     * Gives a string representation fo this client's peronsal information.
	 * @return A string representation of this client's personal information
	 */
	public String toString() {
		return super.toString() + "," + creditCardNumber + "," + expiryDate;
	}

    /**
     * Gets a list of booked itineraries for this client.
     * @return BookedItineraries for this client.
     */
	public List<Itinerary> getBookedItineraries(){
		return bookedItineraries;
	}

}
