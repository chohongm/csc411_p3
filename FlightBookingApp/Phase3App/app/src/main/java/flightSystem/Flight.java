package flightSystem;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.text.DecimalFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

/**
 * Represents a flight with a flights number, departure date and time, arrival
 * date and time, airline, origin, destination, cost and duration.
 * 
 */
public class Flight implements Serializable{
	
	//The flight number of this flight.
	private String flightNumber;
	
	//The departure date and time of this flight in format "yyyy-MM-dd HH:mm".
	private String departureDateTime;
	
	//The arrival date and time of this flight in the format "yyyy-MM-dd HH:mm".
	private String arrivalDateTime;
	
	//The airline of this flight.
	private String airline;
	
	//The origin of this flight.
	private String origin;
	
	//The destination of this flight.
	private String destination;
	
	//The cost of this flight in dollars.
	private double cost;
	
	//The duration of this flight in hours.
	private double duration;

	//The number of seats on this flight.
	private int numSeats;

    //The number of unbooked seats on this flight.
	private int numAvailableSeats;

    //The clients booked on this flight.
	private List<Client> bookedClients;

	/**
	 * Create a new flight with the given information.
	 * @param flightNumber The flight number of this flight.
	 * @param departureDateTime The departure date and time of this flight in
	 * the format "yyyy-MM-dd HH:mm".
	 * @param arrivalDateTime The arrival date and time of this flight in 
	 * the format "yyyy-MM-dd HH:mm".
	 * @param airline The airline of this flight.
	 * @param origin The origin of this flight.
	 * @param destination The destination of this flight.
	 * @param cost The cost of this flight.
     * @param numSeats The number of seats in this flight.
	 * @throws ParseException Throws a ParseException if the any of the flights
	 * have dates that are not formatted as "yyyy-MM-dd HH:mm".
	 */
	public Flight(String flightNumber, String departureDateTime, 
			String arrivalDateTime, String airline, String origin,
			String destination, double cost, int numSeats) throws ParseException {
		
		this.flightNumber = flightNumber;
		this.departureDateTime = departureDateTime;
		this.arrivalDateTime = arrivalDateTime;
		this.airline = airline;
		this.origin = origin;
		this.destination = destination;
		this.cost = cost;
        this.numSeats = numSeats;
		this.numAvailableSeats = numSeats;
		this.duration = calculateDuration();
		this.bookedClients = new ArrayList<Client>();
	}
	
	/**
	 * Calculates and returns the travel duration of this flight.
	 * @return the duration of this flight.
	 * @throws ParseException Throws a ParseException if flight
	 * has dates that are not formatted as "yyyy-MM-dd HH:mm".
	 */
	private double calculateDuration() throws ParseException{
		return FlightSystem.getTimeDifference(departureDateTime,
				arrivalDateTime);
	}

	/**
     * Gets the flight number of this flight.
	 * @return The flight number of this flight.
	 */
	public String getFlightNumber() {
		return flightNumber;
	}

	/**
     * Sets the flight number for this flight
	 * @param flightNumber The flight number to set for this flight.
	 */
	public void setFlightNumber(String flightNumber) {
		this.flightNumber = flightNumber;
	}

	/**
     * Gets the departure date and time of this flight.
	 * @return The departure date and time of this flight.
	 */
	public String getDepartureDateTime() {
		return departureDateTime;
	}

	/**
     * Sets the departure date and time of this flight.
	 * @param departureDateTime The departure date and time to set for this 
	 * flight.
	 */
	public void setDepartureDateTime(String departureDateTime)
            throws ParseException {
		this.departureDateTime = departureDateTime;
		this.duration = calculateDuration();
	}

	/**
     * Gets the arrival date and time of this flight.
	 * @return The arrival date and time of this flight.
	 */
	public String getArrivalDateTime() {
		return arrivalDateTime;
	}

	/**
     * Sets the arrival date and time of this flight.
	 * @param arrivalDateTime The arrival date and time to set for this flight.
	 */
	public void setArrivalDateTime(String arrivalDateTime)
            throws ParseException{
		this.arrivalDateTime = arrivalDateTime;
        this.duration = calculateDuration();
	}

	/**
     * Gets the airline of this flight
	 * @return The airline of this flight.
	 */
	public String getAirline() {
		return airline;
	}

	/**
     * Sets the airline of this flight.
	 * @param airline The airline to set for this flight.
	 */
	public void setAirline(String airline) {
		this.airline = airline;
	}

	/**
     * Gets the origin of this flight.
	 * @return The origin location of this flight.
	 */
	public String getOrigin() {
		return origin;
	}

	/**
     * Sets the origin of this flight.
	 * @param origin The origin location to set for this flight.
	 */
	public void setOrigin(String origin) {
		this.origin = origin;
	}

	/**
     * Gets the destination of this flight.
	 * @return The destination location of this flight.
	 */
	public String getDestination() {
		return destination;
	}

	/**
     * Sets the desitination of this flight.
	 * @param destination The destination location to set for this flight.
	 */
	public void setDestination(String destination) {
		this.destination = destination;
	}

	/**
     * Gets the cost of this flight.
	 * @return The cost of this flight.
	 */
	public double getCost() {
		return cost;
	}

	/**
     * Sets the cost of this flight.
	 * @param cost The cost to set for this flight.
	 */
	public void setCost(double cost) {
		this.cost = cost;
	}

	/**
     * Gets the duration of this flight in hours.
	 * @return The duration of this flight.
	 */
	public double getDuration() {
		return duration;
	}

	/**
     * Sets the duration of this flight in hours.
	 * @param duration The duration to set for this flight.
	 */
	public void setDuration(double duration) {
		this.duration = duration;
	}
	
	/**
	 * Gets just the departure date of this flight and not the departure time.
	 * @return The departure date of this flight.
	 */
	public String getDepartureDate(){
		return this.getDepartureDateTime().split(" ")[0];
	}
	
	/**
	 * Gets just the arrival date of this flight and not the arrival time.
	 * @return The arrival date of this flight.
	 */
	public String getArrivalDate(){
		return this.getArrivalDateTime().split(" ")[0];
	}

    /**
     * Books the given client onto this flight.
     * @param client The client to be booked.
     * @throws NoSeatsAvailableException If there are no seats on this flight.
     * @throws ClientAlreadyBookedException If the client is already booked
     * on this flight.
     */
	public void bookClient(Client client)
            throws NoSeatsAvailableException, ClientAlreadyBookedException {

        //Make sure the client is not already booked
		if (bookedClients.contains(client)) {
			throw new ClientAlreadyBookedException(
                    "The given client is already booked for this flight.");

        //Make sure the flight is not fully booked
		} else if (numAvailableSeats == 0) {
			throw new NoSeatsAvailableException(
                    "No seat is available for this flight.");


        //Make the booking
        } else {
			bookedClients.add(client);
			numAvailableSeats--;
		}
	}

    /**
     * Cancel the client's booking on this flight if it exists.
     * @param client The client to cancel.
     */
	public void cancelBookedClient(Client client) {
		if (bookedClients.remove(client)) {
            numAvailableSeats++;
        }
	}

    /**
     * Gets the booked clients of this flight.
     * @return The bookClient for this flight.
     */
	public List<Client> getBookedClients() {
		return bookedClients;
	}

	/**
	 * Gets the number of seats for this flight.
	 * @return
	 */
	public int getNumAvailableSeats() {
		return numAvailableSeats;
	}

	/**
	 * Sets the number of flights for this flight.
	 * @param numSeats The number of seats for this flight.
	 */
	public void setNumSeats(int numSeats){
		this.numSeats = numSeats;
	}

    /**
     * Gets the number of seats on this flight.
     * @return The number of seats on this flight.
     */
	public int getNumSeats() { return this.numSeats;}

    /**
     * Gives a string representation of a this flights information separated
     * by commas.
	 * @return A string representation of all of this flights information.
	 */
	public String toString() {
		DecimalFormat df = new DecimalFormat("0.00");
		return flightNumber + "," + departureDateTime + "," + arrivalDateTime +
				"," + airline + "," + origin + "," + destination + "," + 
				df.format(cost);
	}
	
	/**
	 * A to string method for displaying itineraries which do not include
     * the individual cost of a flight.
	 * @return A string representation of this Flight without the cost.
	 */
	public String toStringNoCost (){
		return flightNumber + "," + departureDateTime + "," + arrivalDateTime +
				"," + airline + "," + origin + "," + destination;
	}

    /**
     * Compares two flight for equality.
     * @param other The object to compare this flight to.
     * @return True if and only if other is a Flight and has the same
     * flight number and airline as this flight.
     */
	public boolean equals(Object other){
		return (other instanceof Flight)
				&& ((Flight) other).getFlightNumber().equals(this.flightNumber)
				&& ((Flight) other).getAirline().equals(this.airline);
	}
}
