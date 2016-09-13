package com.example.eduardoaleixo.phase3app;

import android.content.Intent;
import android.graphics.Typeface;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.List;

import flightSystem.Client;
import flightSystem.Flight;
import flightSystem.FlightSystem;

/**
 * An activity that allows users to search for flights.
 */
public class SearchFlightActivity extends AppCompatActivity {

    /**
     * Initializes the SerchedFlightsActivity layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_search_flight);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

    }

    /**
     * Searches flights based on the editText on page, then starts an activity
     * to show that result.
     * @param view The current activity.
     */
    public void searchFlights(View view){
        EditText origin = (EditText) findViewById(R.id.editText_origin);
        EditText destination =
                (EditText) findViewById(R.id.editText_destination);
        EditText date = (EditText) findViewById(R.id.editText_date);

        FlightSystem.getInstance().searchFlight(origin.getText().toString(),
                destination.getText().toString(), date.getText().toString());

        Intent oldIntent = getIntent();
        Intent intent = new Intent(this, SearchedFlightsDisplay.class);

        intent.putExtra("curUser", oldIntent.getStringExtra("curUser"));
        startActivity(intent); // Starts DisplayActivity.x

    }




}
