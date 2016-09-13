package com.example.eduardoaleixo.phase3app;

import android.content.Intent;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.TextView;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import flightSystem.Client;
import flightSystem.FlightSystem;
import flightSystem.Itinerary;

/**
 * An activity that allows admins to search for clients.
 */
public class ClientSearch extends AppCompatActivity {

    /**
     * Initializes the ClientSearch layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_client_search);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
    }

    public void searchClientForBooking(View view) {
        int itinerary;
        Intent intent = getIntent();
        String itineraryStr;

        //Get the position of itinerary to be booked in FlightSystem's
        // searchedItineraries
        itinerary = intent.getIntExtra("itinerary", 0);

        //Get a string represention of the itinerary to be booked
        itineraryStr = intent.getStringExtra("itineraryStr");

        //Put the itinerary information into a new intent
        intent = new Intent(this, SearchedClientsDisplay.class);
        intent.putExtra("itinerary", itinerary);
        intent.putExtra("itineraryStr", itineraryStr);

        //Get information from the screen
        TextView lastname = (TextView) findViewById(R.id.lastname);
        TextView firstname = (TextView) findViewById(R.id.firstname);
        TextView email = (TextView) findViewById(R.id.email);

        //Search for clients that match any of the given criteria
        FlightSystem fs = FlightSystem.getInstance();
        ArrayList<Client> searchedClients = fs.searchClients(
                lastname.getText().toString(), firstname.getText().toString(),
                email.getText().toString());

        //Add the searched client to the new intent
        intent.putExtra("searchedClients", (Serializable) searchedClients);
        startActivity(intent);
    }
}
