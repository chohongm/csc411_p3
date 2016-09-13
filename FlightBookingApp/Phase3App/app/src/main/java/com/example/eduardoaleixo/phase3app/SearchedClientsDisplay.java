package com.example.eduardoaleixo.phase3app;

import android.content.Intent;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.TextView;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import flightSystem.Client;
import flightSystem.Flight;
import flightSystem.Itinerary;

/**
 * An activity that displays a list of searched for clients.
 */
public class SearchedClientsDisplay extends AppCompatActivity {

    //The intent that created this activity
    private Intent intent;

    /**
     * Initializes the SearchedClientsDisplay layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        intent = getIntent();
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_searched_clients_display);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        displaySearchedClients();

    }

    /**
     * Displays search clients on the screen
     */
    private void displaySearchedClients() {

        //Get searched clients from intent
        final ArrayList<Client> searchedClients =
                (ArrayList<Client>)
                        intent.getSerializableExtra("searchedClients");

        //Get the itinerary to be booked
        final int itinerary = intent.getIntExtra("itinerary", 0);
        final String itineraryStr = intent.getStringExtra("itineraryStr");

        //Create string representation of the searched clients
        final ArrayList<String> searchedClientsStr = new ArrayList<String>();
        String temp = "";
        for (Client c : searchedClients) {
            temp = String.format(
                    "Lastname: %s\nFirstname: %s\nEmail/Username: %s\nAddress: " +
                            "%s\nCredit Card Number: %s\nExpiry Date: %s\n",
                    c.getLastName(), c.getLastName(), c.getEmail(),
                    c.getAddress(), c.getCreditCardNumber(), c.getExpiryDate());
            searchedClientsStr.add(temp);
        }

        ArrayAdapter<String> arrayAdapter = new ArrayAdapter<String>(this,
                R.layout.client_list_item, android.R.id.text1, searchedClientsStr);

        // create a new ListView, set the adapter and item click listener
        //ListView lv = new ListView(this);
        ListView lv = (ListView) findViewById(R.id.searchedClients);
        lv.setAdapter(arrayAdapter);
        lv.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view,
                                    int position, long id) {

                intent = new Intent(
                        SearchedClientsDisplay.this, ItineraryBook.class);

                //Based on item clicked add info to intent
                intent.putExtra(
                        "curUser", searchedClients.get(position).getEmail());
                intent.putExtra("itinerary", itinerary);
                intent.putExtra("itineraryStr", itineraryStr);

                startActivity(intent);
            }
        });
    }
}
