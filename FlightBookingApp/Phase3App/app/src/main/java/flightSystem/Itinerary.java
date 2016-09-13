package flightSystem;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Represents a sequence of flights from origin to destination 
 * that make up an itinerary. 
 *
 */
public class Itinerary implements Serializable{
	
	//The flights that make up this itinerary, listed in travel order.
	private List<Flight> flightSequence = new ArrayList<Flight>();
	
	//The total cost of this whole itinerary.
	private double totalCost;
	
	//The total travel time of this whole itinerary.
	private double travelTime;

    /**
	 * Initialize an empty Itinerary.
	 */
	public Itinerary(){
		totalCost = 0;
		travelTime = 0;
	}
	
	/**
	 * Initialize a new Itinerary with the given list of flights that 
	 * represent a flight sequence.
	 * @param flights The flight sequence of the new Itinerary.
	 * @throws ParseException Throws a ParseException if the any of the flights
	 * have dates that are not formatted as "yyyy-MM-dd HH:mm".
	 */
	public Itinerary(List<Flight> flights) throws ParseException{
		this.flightSequence = flights;
		calculateTotalCost();
		calculateTravelTime();
	}

    /**
     * Checks whether or not all flights in this itinerary have seats available.
     * @return True if and only if all flight in this itinerary have seats
     * available.
     */
	public boolean allFlightsAvailable() {
		for (Flight f : flightSequence) {
			if (f.getNumAvailableSeats() == 0) {
				return false;
			}
		}
		return true;
	}
	
	/**
	 * Adds a flight to the flight sequence in this Itinerary and updates 
	 * relevant fields.
	 * @param flight The flight to be added to the flight sequence.
	 * @throws ParseException Throws a ParseException if the any of the flights
	 * have dates that are not formatted as "yyyy-MM-dd HH:mm".
	 */
	public void addFlight(Flight flight) throws ParseException{
		if (this.flightSequence.size() >= 1) {
			
			//temporarily store last flight for comparison
			int length = this.flightSequence.size();
			Flight f = this.flightSequence.get(length-1);
			
			//check if origin/destination is correct, check layover time
			if (flight.getOrigin() == f.getDestination() 
				&& FlightSystem.getTimeDifference(f.getArrivalDateTime(), 
						flight.getDepartureDateTime())<= 6.0) {
				this.flightSequence.add(flight);
				//update itinerary cost and time
				calculateTotalCost();
				calculateTravelTime();
			}
		}
		
		else {
			this.flightSequence.add(flight);
			
			//update itinerary cost and time
			calculateTotalCost();
			calculateTravelTime();
		}
	}
	
	/**
	 * Calculate the total cost of this itinerary and update the totalCost 
	 * field.
	 */
	private void calculateTotalCost(){
		totalCost = 0;
		
		//Adds up the cost of the whole itinerary.
		for (int i = 0; i < flightSequence.size(); i++){
			totalCost = totalCost + flightSequence.get(i).getCost();
		}
	}
	
	/**
	 * Calculate the total travel time of this itinerary and update the 
	 * travelTime field.
	 * @throws ParseException Throws a ParseException if the any of the flights
	 * have dates that are not formatted as "yyyy-MM-dd HH:mm".
	 */
	private void calculateTravelTime() throws ParseException{
		
		//Gets the departure time and arrival time of the whole itinerary.
		int length = flightSequence.size();
		String departure = flightSequence.get(0).getDepartureDateTime();
		String arrival = flightSequence.get(length-1).getArrivalDateTime();
				
		//Calculate the total travel time of the whole itinerary.
		travelTime = FlightSystem.getTimeDifference(departure, arrival);
	}

	
	/**
     * Gets the flight sequence of this itinerary.
	 * @return The sequence of flights in this itinerary.
	 */
	public List<Flight> getFlightSequence() {
		return flightSequence;
	}
	
	/**
     * Sets the flight sequence
	 * @param flightSequence The sequence of flights to set for this itinerary.
	 */
	public void setFlightSequence(List<Flight> flightSequence) {
		this.flightSequence = flightSequence;
	}
	
	/**
     * Gets the total cost of this itinerary.
	 * @return The total cost of this itinerary.
	 */
	public double getTotalCost() {
		return totalCost;
	}
	
	/**
     * Sets the total cost of this itinerary.
	 * @param totalCost The total cost to set for this itinerary.
	 */
	public void setTotalCost(double totalCost) {
		this.totalCost = totalCost;
	}
	
	/**
     * Gets the total travel time for this itinerary in hours.
	 * @return The total travel time for this itinerary.
	 */
	public double getTravelTime() {
		return travelTime;
	}
	
	/**
     * Sets the travel time for this itinerary in hours.
	 * @param travelTime The total travel time to set for this itinerary.
	 */
	public void setTravelTime(double travelTime) {
		this.travelTime = travelTime;
	}

	/**
     * Gives a string representaiton of this itinerary.
	 * @return A string representation of this itinerary in the format:
	 * Number,DepartureDateTime,ArrivalDateTime,Airline,Origin,Destination 
	 * total price (two decimal places)
	 * total duration (Formatted as HH:MM).
	 */
	public String toString() {
        if (flightSequence.size() == 0) {
            return "";
        }
        DecimalFormat df = new DecimalFormat("#.00");
        String ret = "";
        for (int i = 0; i < flightSequence.size(); i++) {
            ret += flightSequence.get(i).toStringNoCost();
            ret += "\n";
        }

        ret += df.format(totalCost) + "\n";

        int hours = (int) Math.floor(travelTime);
        int minutes = (int) Math.round(((travelTime - hours) * 60));
        String result = String.format("%02d", hours) + ":" +
                String.format("%02d", minutes);
        ret += result;
        return ret;
    }

    /**
     * Compares an object to this object for equality.
     * @param o The object to compare.
     * @return True if and only if other is an itinerary and all flights in
     * other's flight sequence are equal to this flight sequence.
     */
    @Override
    public boolean equals(Object o) {
        if (o instanceof Itinerary &&
                ((Itinerary) o).getFlightSequence().size()
                        == this.flightSequence.size()){

            for (int i = 0; i < flightSequence.size(); i++){
                if (!((Itinerary) o).getFlightSequence().get(i).
                        equals(this.flightSequence.get(i))) {

                    return false;
				}
            }
			return true;
        }
        return false;
    }

    /**
     * Gives are string representation of this itinerary for use in the app.
     * @return A string representation fo this itinerary.
     */
	public String toDisplayString(){
		String temp = "";

		//Put headings onto flight info
        for (int f=1; f <= flightSequence.size(); f++) {
			String[] flightInfo =
                    flightSequence.get(f - 1).toString().split(",");

            temp += "\nFlight" + f + "\n" + "Flight Number: " + flightInfo[0] +
                    "\nDeparture Datetime: " + flightInfo[1] +
                    "\nArrival Datetime: " + flightInfo[2] +
					"\nAirLine: " + flightInfo[3] + "\nOrigin: " +
                    flightInfo[4] + "\nDestination: " + flightInfo[5] +
                    "\nCost: " + flightInfo[6] + "\n";
		}

        //Format the cost to have two decimal places
		DecimalFormat df = new DecimalFormat("#.00");
		String totalCost = df.format(this.getTotalCost()) + "\n";

        //Format the total travel time into the form HH:MM
        Double totalTravelTime = this.getTravelTime();
		int hours = (int) Math.floor(totalTravelTime);
		int minutes = (int) Math.round(((totalTravelTime - hours) * 60));
		String totalDuration = String.format("%02d", hours) + ":" +
				String.format("%02d", minutes);

        //Put together the whole string
		temp += "\nTotal Duration: " + totalDuration + "\nTotal Cost: " +
                totalCost + "\n";

		return temp;
	}
}
