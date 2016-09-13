package com.example.eduardoaleixo.phase3app;

import android.app.AlertDialog;
import android.content.Intent;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.app.Activity;
import android.widget.SimpleAdapter;
import android.widget.TextView;
import android.widget.Toast;

import java.io.FileNotFoundException;
import java.io.Serializable;
import java.lang.reflect.Array;
import java.text.DecimalFormat;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Handler;

import flightSystem.Client;
import flightSystem.FileFormatException;
import flightSystem.Flight;
import flightSystem.FlightSystem;
import flightSystem.Itinerary;

/**
 * An activity that displays a list of searched for itineraries.
 */
public class SearchedItinerariesDisplay extends AppCompatActivity {

    // singleton instance of FlightSystem
    private Intent intent;
    // current user instance using this activity
    private String curUser;
    // singleton instance of FlightSystem
    private FlightSystem fs;

    /**
     * Initializes the SerchedItinerariesDisplay layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_searched_itineraries_display);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        intent = getIntent();
        curUser = intent.getStringExtra("curUser");

        fs = FlightSystem.getInstance();

        displaySearchedItins();
    }

    /**
     * Displays searched itineraries in string format and upon clicking one of
     * the displayed itineraries, the method sends the selected itinerary,
     * a string representing the selected itinerary, as well as current user
     * in new intent to the next activity.
     */
    public void displaySearchedItins() {
        final List<Itinerary> searchedItins = fs.getSearchedItineraries();
        final List<String> searchedItinsStr = new ArrayList<String>();

        for (Itinerary i : searchedItins) {
            if (i.allFlightsAvailable()) {
                searchedItinsStr.add(i.toDisplayString());
            }
        }

        //This is the array adapter, it takes the context of the activity as a
        //first parameter, the type of list view as a second parameter and your
        //array as a third parameter.
        ArrayAdapter<String> arrayAdapter = new ArrayAdapter<String>(this,
                android.R.layout.simple_list_item_1,
                searchedItinsStr);

        //Create a new ListView, set the adapter and item click listener
        //ListView lv = new ListView(this);
        ListView lv = (ListView) findViewById(R.id.searchedItins);
        lv.setAdapter(arrayAdapter);
        lv.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Intent intent;

                //Sends admins to a different booking page that allows them
                // to enter the client they want to book for.
                if (curUser == null){
                    intent = new Intent(SearchedItinerariesDisplay.this,
                            ClientSearch.class);
                } else {
                    intent = new Intent(SearchedItinerariesDisplay.this,
                            ItineraryBook.class);
                }

                intent.putExtra("itinerary", position);
                intent.putExtra("itineraryStr", searchedItinsStr.get(position));
                intent.putExtra("curUser", curUser);
                //based on item add info to intent
                startActivity(intent);
            }


        });
    }

}

