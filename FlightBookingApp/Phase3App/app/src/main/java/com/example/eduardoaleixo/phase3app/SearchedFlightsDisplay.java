package com.example.eduardoaleixo.phase3app;

import android.content.Intent;
import android.graphics.Typeface;
import android.os.Bundle;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.Button;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import flightSystem.Client;
import flightSystem.Flight;
import flightSystem.FlightSystem;

/**
 * An activity that displays a list of searched for flights.
 */
public class SearchedFlightsDisplay extends AppCompatActivity
        implements View.OnClickListener{

    /**
     * Initializes the SerchedFlightsDisplay layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_searched_flights_display);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        //Get the searched flights from the FlightSystem
        FlightSystem fs = FlightSystem.getInstance();
        final List<Flight> searchResult = fs.getSearchedFlights();

        init(searchResult);
    }

    /**
     * Shows the flights searched on screen.
     * @param searchResult  A list of Flights, which are displayed on screen.
     */
    private void init(List<Flight> searchResult){
        TableLayout table = (TableLayout) findViewById(R.id.flightsTable);
        table.removeAllViews();

        TableRow row = new TableRow(this);
        TextView label;
        TextView content;
        int i = 0;

        // if there are no flights stored
        if (searchResult.size() <= 0){
            table.addView(createFlightColumn("No flights found.", ""));
        }

        table.addView(createFlightColumn("Number of flights found: ",
                searchResult.size() + ""));
        table.addView(createFlightColumn("", ""));


        for (Flight flight : searchResult) {
            DecimalFormat df = new DecimalFormat("0.00");

            table.addView(createFlightColumn(
                    "" + R.string.edit_flightNumber, flight.getFlightNumber()));
            table.addView(createFlightColumn(
                    "" + R.string.edit_airline, flight.getAirline()));
            table.addView(createFlightColumn(
                    "" + R.string.edit_origin, flight.getOrigin()));
            table.addView(createFlightColumn(
                    "" + R.string.edit_destination, flight.getDestination()));
            table.addView(createFlightColumn(
                    "" + R.string.edit_departure, flight.getDepartureDateTime()));
            table.addView(createFlightColumn(
                    "" + R.string.flight_duration, df.format(flight.getDuration())));
            table.addView(createFlightColumn(
                    "" + R.string.edit_cost, df.format(flight.getCost())));
            table.addView(createFlightColumn(
                    "" + R.string.available_seats, flight.getNumAvailableSeats() + ""));


            Intent intent = getIntent();

            // If there's no current user, then it's an admin who can edit
            // flights
            if (intent.getStringExtra("curUser") == null){
                Button b = new Button(this);
                b.setTag(flight);
                b.setText(R.string.edit_flight_info);
                b.setOnClickListener(this);
                table.addView(b);
            }


            table.addView(createFlightColumn("", ""));
            table.addView(createFlightColumn("", ""));
        }
    }

    /**
     * Handles the click on "Edit this flight" buttons.
     * @param v The view which corresponds to this activity.
     */
    public void onClick(View v){
        Flight f = (Flight) v.getTag();
        TableLayout table = (TableLayout) findViewById(R.id.flightsTable);
        table.addView(createFlightColumn("f", f.toString()));

        Intent intent = new Intent(this, EditFlightActivity.class);
        intent.putExtra("flight", (Serializable) f);
        startActivity(intent); // Starts DisplayActivity.x
    }

    /**
     * Creates a formated row to be added to a table.
     * @param label The label to the row.
     * @param content The content of the row.
     * @return A TableRow, which can be added to a TableLayout.
     */
    private TableRow createFlightColumn(String label, String content){
        //Creates a row, where one column is the label and the other the content
        TableRow row = new TableRow(this);
        TextView tv_label = new TextView(this);
        tv_label.setTypeface(null, Typeface.BOLD);
        TextView tv_content = new TextView(this);

        tv_label.setText(label);
        tv_content.setText(content);
        row.addView(tv_label);
        row.addView(tv_content);

        return row;
    }
}
