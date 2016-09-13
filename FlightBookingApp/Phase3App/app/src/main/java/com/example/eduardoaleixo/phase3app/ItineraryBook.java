package com.example.eduardoaleixo.phase3app;

import android.content.Intent;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.TextView;

import java.io.IOException;
import java.io.Serializable;

import flightSystem.Admin;
import flightSystem.Client;
import flightSystem.ClientAlreadyBookedException;
import flightSystem.FlightSystem;
import flightSystem.Itinerary;
import flightSystem.NoSeatsAvailableException;
import flightSystem.User;

/**
 * An activity that allows users to book itineraries.
 */
public class ItineraryBook extends AppCompatActivity {

    //The backend FlightSystem instance
    private FlightSystem fs;

    //The user to be booked
    private Client curUser;

    //The position of the itinerary to be booked in the FlightSystem
    // bookedItineraries
    private int itinerary;

    /**
     * Initializes the ItineraryBook layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Intent intent;

        //String representation of the itinerary to be booked
        String itineraryStr;

        //Get the intent that created this activity and it's extras
        intent = getIntent();
        itinerary = intent.getIntExtra("itinerary", 0);
        itineraryStr = intent.getStringExtra("itineraryStr");

        //Find the client that corresponds with the email from the intent
        fs = FlightSystem.getInstance();
        curUser = fs.searchClient(intent.getStringExtra("curUser"));

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_itinerary_book);

        //Display the itinerary to be booked
        TextView itineraryTV = (TextView) findViewById(R.id.itinerary);
        itineraryTV.setText(itineraryStr);

        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
    }

    /**
     * Starts SearchFlightActivity activity when the relevant button is pressed.
     * @param view The view of this activity.
     */
    public void bookItinerary(View view) {
        TextView error_message = (TextView) findViewById(R.id.booking_message);

        try {
            //Book the itinerary
            curUser.bookItinerary(fs.getSearchedItineraries().get(itinerary));

            //Save the booking
            fs.save(this.getApplicationContext().getFilesDir().toString());
            error_message.append(curUser.getEmail() + " is booked successfully.");

        } catch (NoSeatsAvailableException e) {
            error_message.setText(R.string.flight_full);
        } catch (IOException e) {
            error_message.setText(R.string.booking_not_saved);
        } catch (ClientAlreadyBookedException e) {
            error_message.setText(R.string.client_already_booked);
        }
    }
}
