package flightSystem;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.FileNotFoundException;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;


import java.util.Comparator;

/**
 * The bulk of the backend for the flight booking system. Loads, saves and
 * and stores all the data needed for the application. Also performs and stores
 * the results of searches.
 *
 */
public class FlightSystem implements Serializable{

    //For serialization
    private static final long serialVersionUID = 1L;

    //The list of flights for this system.
    private List<Flight> flights;

    //The list of clients that use this system.
    private List<Client> clientList;

    //The results of the last itinerary search.
    private List<Itinerary> searchedItineraries;

    //The results of the last flight search.
    private List<Flight> searchedFlights;

    //Constants representing the number of values that should be in the csv
    //files.
    private static final int NUM_FLIGHT_PARAMS = 8;
    private static final int NUM_CLIENT_PARAMS = 6;

    private static FlightSystem singletonSystem;

    /**
     * Creates a new flight system with no data in it.
     */
    private FlightSystem() {
        flights = new ArrayList<Flight>();
        clientList = new ArrayList<Client>();
        searchedItineraries = new ArrayList<Itinerary>();
        searchedFlights = new ArrayList<Flight>();
    }

    /**
     * Singleton Constructor.
     * @return A Singleton representation of the System.
     */
    public static FlightSystem getInstance(){
        if (singletonSystem == null){
            singletonSystem = new FlightSystem();
        }

        return singletonSystem;
    }

    /**
     * Load a FlightSystem object from file
     * @param filePath The path to the save.obj file.
     */
    public static void load(String filePath){
        FileInputStream fis;
        ObjectInputStream ois;

        try {
            fis = new FileInputStream(filePath + "/save.obj");
            ois = new ObjectInputStream(fis);

            singletonSystem = (FlightSystem) ois.readObject();

            ois.close();
            fis.close();
        } catch (ClassNotFoundException e) {
            singletonSystem = new FlightSystem();
        } catch (IOException e) {
            singletonSystem = new FlightSystem();
        }
    }

    /**
     * Serialize this FlightSystem.
     * @param filePath The path to the folder that save.obj will be written to.
     * @throws IOException If the save is not successful.
     */
    public void save(String filePath) throws IOException {
        FileOutputStream fos  = new FileOutputStream(filePath + "/save.obj");
        ObjectOutputStream oos  = new ObjectOutputStream(fos);

        oos.writeObject(this);

        oos.close();
        fos.close();
    }

    /**
     * Searches for and returns all direct flights that match the given flight
     * information.
     * @param origin The desired origin for the flights being searched.
     * @param destination The desired destination of the flights being searched.
     * @param date The desired departure date of flights being searched.
     * @return A list of all flights that match the given flight info.
     * An empty List in case of no flight found.
     */
    public List<Flight> searchFlight(String origin, String destination,
                                     String date){

        List<Flight> retFlights = new ArrayList<Flight>();

        for (Flight f1 : flights) {
            if (f1.getOrigin().equals(origin)
                    && f1.getDepartureDate().equals(date)
                    && f1.getDestination().equals(destination)) {
                retFlights.add(f1);
            }
        }

        searchedFlights = retFlights;
        return retFlights;
    }

    /**
     * Searches for ONE flight in the system, given a Flight object.
     * @param flight The Flight which should be looked for in the System.
     * @return Exactly ONE Flight object.
     */
    public Flight searchOneFlight(Flight flight){
        Flight f = null;
        for (Flight f1 : flights) {
            if (f1.getFlightNumber().equals(flight.getFlightNumber())
                    && f1.getDepartureDateTime().equals(
                    flight.getDepartureDateTime())
                    && f1.getAirline().equals(flight.getAirline())
                    && f1.getOrigin().equals(flight.getOrigin())
                    && f1.getDestination().equals(flight.getDestination())) {
                f = f1;
            }
        }

        return f;
    }

    /**
     * Searches for and returns all itineraries that match the given origin,
     * final destination, and departure date.
     * @param origin The desired origin of the itineraries being searched.
     * @param destination The desired final destination of the itineraries
     * being searched.
     * @param date The desired departure date of itineraries being searched.
     * @return A list of all itineraries that match the given itinerary info.
     * @throws ParseException Throws a ParseException if the any of the flights
     * have dates that are not formatted as "yyyy-MM-dd HH:mm".
     */
    public List<Itinerary> searchForItineraries (
            String origin, String destination, String date)
            throws ParseException {

        List<Itinerary> itineraries = new ArrayList<Itinerary>();
        for (Flight f1 : flights) {
            if (f1.getOrigin().toLowerCase().equals(origin.toLowerCase())
                    & f1.getDepartureDate().equals(date)) {

                // Direct flight itineraries.
                if (f1.getDestination().toLowerCase().equals(
                        destination.toLowerCase())) {
                    List<Flight> flightSequence = new ArrayList<Flight>();
                    flightSequence.add(f1);
                    itineraries.add(new Itinerary(flightSequence));

                    // Indirect flight itineraries.
                } else {

                    // Recursively find all possible flight sequences that
                    // include f1 as the 1st flight in each.
                    List<List<Flight>> flightSequences = getFlightSequences(
                            flights, f1, destination, origin);
                    // Create an itinerary for each returned flight sequence
                    // and add it to itineraries list.
                    for (List<Flight> fs : flightSequences) {
                        itineraries.add(new Itinerary(fs));
                    }
                }
            }
        }
        searchedItineraries = itineraries;
        return searchedItineraries;
    }


    /**
     * Recursively finds and returns all possible flight sequences that starts
     * with the given flight as the 1st flight in each sequence, and ends with
     * a flight whose destination is equal to the given final destination.
     * Helper method for searchForItineraries.
     * @param flights List of all flights that can be added to each flight
     * sequence.
     * @param f1 The first flight in each flight sequence being created.
     * @param destination The final destination of each flight sequence.
     * @param originOfPreviousFlight The origin of the previous flight in the
     * flight sequence.
     * @return The list of flight sequences that match the given flight
     * sequence info.
     * @throws ParseException Throws a ParseException if the any of the flights
     * have dates that are not formatted as "yyyy-MM-dd HH:mm".
     */
    private List<List<Flight>> getFlightSequences (
            List<Flight> flights, Flight f1, String destination,
            String originOfPreviousFlight)
            throws ParseException {

        // Create a copy of flights for recursive call.
        List<Flight> flightsCopy = new ArrayList<Flight>(flights);

        // Remove any flight that has destination equal to origin of the
        // previous connecting flight.
        // This prevents the itinerary being created from revisiting
        // the same place in the flight sequence.
        List<Flight> flightsToRemove = new ArrayList<Flight>();
        for (Flight f : flightsCopy) {
            if (f.getDestination().toLowerCase().equals(
                    originOfPreviousFlight.toLowerCase())) {

                flightsToRemove.add(f);
            }
        }
        for (Flight f : flightsToRemove) {
            flightsCopy.remove(f);
        }
        List<List<Flight>> flightSequences = new ArrayList<List<Flight>>();

        // Search for each flight that has origin equal to arrival of the
        // previous flight and has stop over time within 6 hours.
        for (Flight f2 : flightsCopy) {

            double timeDifference = getTimeDifference(f1.getArrivalDateTime(),
                    f2.getDepartureDateTime());

            if (f2.getOrigin().toLowerCase().equals(
                    f1.getDestination().toLowerCase())
                    & timeDifference <= 6 & timeDifference >= 0) {

                // If the flight (f2) arrives at the final destination,
                // add the flight sequence f1 -> f2 to flight sequences.
                if (f2.getDestination().toLowerCase().equals(
                        destination.toLowerCase())) {

                    List<Flight> flightSequence = new ArrayList<Flight>();
                    flightSequence.add(f1);
                    flightSequence.add(f2);
                    flightSequences.add(flightSequence);
                }

                // If f2 does not arrive at the final destination,
                // get the next all possible flight sequences recursively
                // and then add flight sequences f1 -> each returned flight
                // sequence from f2 to flight sequences.
                else {
                    List<List<Flight>> retFlightSequences =
                            getFlightSequences(flightsCopy, f2, destination,
                                    f2.getOrigin());
                    for (List<Flight> fs : retFlightSequences) {
                        List<Flight> flightSequence = new ArrayList<Flight>();
                        flightSequence.add(f1);
                        flightSequence.addAll(fs);
                        flightSequences.add(flightSequence);
                    }
                }
            }
        }
        return flightSequences;
    }

    /**
     * Calculates and returns the time difference between two times.
     * Parameters must be in the format "yyyy-MM-dd HH:mm".
     * @param date1 The first date and time in "yyyy-MM-dd HH:mm" format.
     * @param date2 The second date and time in "yyyy-MM-dd HH:mm" format.
     * @return The time difference in hours between date1 and date2
     * @throws ParseException Throws a ParseException if the dates are given
     * in the wrong format.
     */
    public static double getTimeDifference(String date1, String date2)
            throws ParseException {

        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd HH:mm");
        Date d1 = null;
        Date d2 = null;
        double diffHours = 0;

        d1 = format.parse(date1);
        d2 = format.parse(date2);

        //Calculate the time difference in milliseconds.
        double diff = d2.getTime() - d1.getTime();

        //Convert the time difference to hours.
        diffHours = diff / (60 * 60 * 1000);

        return diffHours;
    }

    /**
     * Adds the flights found in filename to this system's flights list.
     * @param filename the path to a file of flight information where the lines
     *            must be in the format:
     *            Number,DepartureDateTime,ArrivalDateTime,Airline,Origin,
     *            Destination,Price (the dates and times are in the format
     *            YYYY-MM-DD HH:MM)
     * @throws FileNotFoundException Throws a FileNotFoundException if the
     * filename path does not exist.
     * @throws ParseException Throws a ParseException if any flight has dates
     * that are not formatted as "yyyy-MM-dd HH:mm".
     * @throws FileFormatException Throws a FileFormatException if the given
     * filename does not have the right number of values or if the cost is
     * not a valid number.
     */
    public void loadFlights(String filename) throws FileNotFoundException,
            ParseException, FileFormatException{

        //Open the specified file
        Scanner sc = new Scanner(new File(filename));

        //Temporary variables for reading the file.
        String nextLine;
        String [] flightData;

        //Cycle through each line of the file
        while (sc.hasNextLine()){
            nextLine = sc.nextLine();
            flightData = nextLine.split(",");

            //Makes sure that the file is in the right format.
            if (flightData.length != NUM_FLIGHT_PARAMS){
                throw new FileFormatException("Wrong number of values");
            }

            //Add the flight while making sure that the second last value is
            //actually a cost and the last value is a number of seats.
            try{
                addFlight(flightData[0], flightData[1],
                        flightData[2], flightData[3], flightData[4],
                        flightData[5], Double.parseDouble(flightData[6]),
                        Integer.parseInt(flightData[7]));
            }catch (NumberFormatException ex){
                throw new FileFormatException(
                        "Cost must be a number/NumSeats must be an Integer");
            }
        }
    }

    /**
     * Adds a flight with the given information to this system's flight list.
     * @param number The ID number of the flight.
     * @param departure The departure date and time of the flight in the format
     * "yyyy-MM-dd HH:mm".
     * @param arrival The arrival date and time of the flight in the format
     * "yyyy-MM-dd HH:mm".
     * @param airline The airline of the flight.
     * @param origin The origin of the flight.
     * @param destination The destination of the flight.
     * @param price The price of the flight
     * @param numSeats The number of seats in the flight.
     * @throws ParseException Throws a ParseException if the departure and/or
     * arrival dates are not in the format "yyyy-MM-dd HH:mm".
     */
    public void addFlight(String number, String departure, String arrival,
                          String airline, String origin, String destination,
                          double price, int numSeats)
            throws ParseException{

        Flight newFlight = new Flight(number, departure, arrival, airline,
                origin, destination, price, numSeats);

        // arrival time is earlier than departure
        if (getTimeDifference(departure, arrival) < 0){
            return ;
        }

        //Subsequent uploads of flights with the same flightNumber and
        //airline will replace the old instance of the flight
        removeOldFlight(newFlight);

        flights.add(newFlight);
    }

    /**
     * Removes a flight from this flights list if it has the same flightNumber
     * and airline as the newFlight.
     * @param newFlight The flight to compare to for removal.
     */
    private void removeOldFlight(Flight newFlight){
        int toBeRemoved = -1;

        //Find matching flights
        for (int i = 0; i < flights.size(); i++){
            if (flights.get(i).equals(newFlight)){
                toBeRemoved = i;
            }
        }

        //Remove matching flight if it was found
        if (toBeRemoved != -1) {
            flights.remove(toBeRemoved);
        }
    }

    /**
     * Adds the clients found in filename to this systems list of clients.
     * @param filename the path to a file of client information where the lines
     *            must be in the format:
     *            LastName,FirstNames,Email,Address,CreditCardNumber,ExpiryDate
     *            (ExpiryDate is stored in the format YYYY-MM-DD)
     * @throws FileNotFoundException Throws a FileNotFoundException if the
     * filename path does not exist.
     * @throws FileFormatException Throws a FileFormatException if the given
     * filename does not have the right number of values.
     */
    public void loadClients(String filename) throws FileNotFoundException,
            FileFormatException{

        //Opens the specified file.
        Scanner sc = new Scanner(new File(filename));

        //Temporary variables for reading the file.
        String nextLine;
        String [] clientData;

        while (sc.hasNextLine()){
            nextLine = sc.nextLine();
            clientData = nextLine.split(",");

            //Makes sure that the file is in the right format
            if (clientData.length != NUM_CLIENT_PARAMS){
                throw new FileFormatException();
            }
            addClient(clientData[0], clientData[1],
                    clientData[2], clientData[3], clientData[4],
                    clientData[5]);
        }
    }

    /**
     * Adds a client with the given information to this system's client list.
     * @param lastName The last name of the client.
     * @param firstName The first name of the client.
     * @param email The email of the client.
     * @param address The address of the client.
     * @param creditCardNumber The credit card number of the client.
     * @param expiryDate The expiry date of the client's credit card.
     */
    public void addClient(String lastName, String firstName, String email,
                          String address, String creditCardNumber,
                          String expiryDate){

        Client c = null;

        //Do not add clients with completely empty info
        if (lastName.length() == 0 && firstName.length() == 0
                && address.length() == 0 && creditCardNumber.length() == 0
                && expiryDate.length() == 0){

            return;
        }

        //Override existing client info if a client with the same email is
        // already in the system
        c = this.searchClient(email);
        if (c != null) {
            c.setLastName(lastName);
            c.setFirstName(firstName);
            c.setEmail(email);
            c.setAddress(address);
            c.setCreditCardNumber(creditCardNumber);
            c.setExpiryDate(expiryDate);
        } else {

            //Add the client
            c = new Client(lastName, firstName, email, address,
                    creditCardNumber, expiryDate);
            clientList.add(c);
        }


    }

    /**
     * Sort this system's searched itineraries in ascending order by cost.
     * @return This system's searched itineraries sorted in ascending order
     * by cost.
     */
    public List<Itinerary> sortItinerariesByCost(){
        Collections.sort(searchedItineraries, new ItinCostComparator());
        return searchedItineraries;

    }

    //A comparator to make sortItinerariesByCost easier to implement
    private class ItinCostComparator implements Comparator<Itinerary> {
        @Override
        public int compare(Itinerary i1, Itinerary i2) {
            if (i1.getTotalCost() > i2.getTotalCost()){
                return 1;
            } else if (i1.getTotalCost() < i2.getTotalCost()){
                return -1;
            } else {
                return 0;
            }
        }
    }

    /**
     * Sort this system's searched itineraries in ascending order by travel
     * time.
     * @return This system's searched itineraries sorted in ascending order
     * by travel time.
     */
    public List<Itinerary> sortItinerariesByTime(){
        Collections.sort(searchedItineraries, new ItinTravelTimeComparator());
        return searchedItineraries;
    }

    //A comparator to make sortItinerariesByTime easier to implement.
    private class ItinTravelTimeComparator implements Comparator<Itinerary> {
        @Override
        public int compare(Itinerary i1, Itinerary i2) {
            if (i1.getTravelTime() > i2.getTravelTime()){
                return 1;
            } else if (i1.getTravelTime() < i2.getTravelTime()){
                return -1;
            } else {
                return 0;
            }
        }
    }

    /**
     * Searches and returns a client that matches given email from client list
     * or throws an exception if not found.
     * @param email an email address of the client to be found.
     * @return the client with given email from client list.
     */
    public Client searchClient(String email) {
        for (Client c : clientList) {
            if (c.getEmail().equals(email)) {
                return c;
            }
        }
        // In case of no Client found
        return null;
    }

    /**
     * Gives a string representation of this clientList.
     * @return A string representation of this system's clientList.
     */
    public String displayClients() {
        String ret = "";

        //Cycle through all clients in this system's client list to create a
        //string representation.
        if (clientList.size() != 0) {
            for (int i=0; i < clientList.size(); i++) {
                ret += clientList.get(i).toString();
                if (i != clientList.size() - 1) {
                    ret += "\n";
                }
            }
        }
        return ret;
    }

    /**
     * Gives a string representation of this searchedItineraries that have
     * seats available for booking.
     * @return A string representation of this system's current
     * searchedItineraries.
     */
    public String displayItineraries(){
        String ret = "";

        //Cycle through the current itinerary search to create a string
        //representation.
        if (searchedItineraries.size() != 0) {
            for (int i=0; i < searchedItineraries.size(); i++) {

                //Only add to the return string if this itinerary is bookable
                // i.e. All its flights have seats.
                if (searchedItineraries.get(i).allFlightsAvailable()) {
                    ret += searchedItineraries.get(i).toString();
                    if (i != searchedItineraries.size() - 1) {
                        ret += "\n";
                    }
                }
            }
        }
        return ret;
    }

    /**
     * Gives a string representation of this searchedFlights.
     * @return A string representation of this system's current
     * searchedFlights.
     */
    public String displaySearchedFlights() {

        String ret = "";

        //Cycle through the current flight search to create a string
        //representation.
        for (int i=0; i < searchedFlights.size(); i++) {
            ret += searchedFlights.get(i).toString();
            if (i != searchedFlights.size() - 1) {
                ret += "\n";
            }
        }
        return ret;
    }


    /**
     * Searches for clients in this clientList that match at least one of the
     * given parameters.
     * @param lastname Last name to search for.
     * @param firstname First name to search for.
     * @param email Email to search for.
     * @return A list of clients that match any of three given parameters.
     */
    public ArrayList<Client> searchClients(String lastname, String firstname, String email) {
        List<Client> searchedClients = new ArrayList<Client>();

        //Cycle through this client list to perfom a linear search
        for (Client c : clientList) {
            if (c.getLastName().toLowerCase().equals(lastname.toLowerCase())
                    || c.getFirstName().toLowerCase().equals(
                    firstname.toLowerCase())
                    || c.getEmail().toLowerCase().equals(email.toLowerCase())) {
                searchedClients.add(c);
            }
        }
        return (ArrayList) searchedClients;
    }

    /**
     * Destroys the system Singleton,
     * setting it to null.
     */
    public void destroy(){

        // Used in test cases
        this.singletonSystem = null;
    }

    /**
     * Gets a list of all clients.
     * @return A List of Clients
     */
    public List<Client> getClients() {
        return this.clientList;
    }

    /**
     * Gets a list of all flights.
     * @return A list of Flights.
     */
    public List<Flight> getFlights(){
        return this.flights;
    }

    /**
     * Gets a list of the current searched for itineraries.
     * @return A list of searched for itineraries.
     */
    public List<Itinerary> getSearchedItineraries (){
        return searchedItineraries;
    }

    /**
     * Gets a list of the current searched for flights.
     * @return A list of searched for flights.
     */
    public List<Flight> getSearchedFlights(){ return searchedFlights; }

}
