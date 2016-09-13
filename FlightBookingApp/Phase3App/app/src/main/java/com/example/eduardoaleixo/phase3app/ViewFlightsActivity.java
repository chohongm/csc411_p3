package com.example.eduardoaleixo.phase3app;

import android.graphics.Typeface;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import java.text.DecimalFormat;

import flightSystem.Flight;
import flightSystem.FlightSystem;

/**
 * An activity that allows an admin to view all of the flights in the system.
 */
public class ViewFlightsActivity extends AppCompatActivity {

    /**
     * Initializes the ViewFlightsActivity layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_view_flights);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        init();
    }

    /**
     * Creates a formated row to be added to a table.
     * @param label The label to the row.
     * @param content The content of the row.
     * @return A TableRow, which can be added to a TableLayout.
     */
    private TableRow createFlightColumn(String label, String content){
        // creates a row, where one column is the label
        // and the other the content
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

    /**
     * Creates elements to display all flights available on the system.
     */
    private void init(){
        TableLayout table = (TableLayout) findViewById(R.id.flightsTable);

        TableRow row = new TableRow(this);
        TextView label;
        TextView content;

        // if there are no flights stored
        if (FlightSystem.getInstance().getFlights().size() <= 0){
            table.addView(createFlightColumn("No flights available.", ""));
        }

        for (Flight flight : FlightSystem.getInstance().getFlights()){
            DecimalFormat df = new DecimalFormat("0.00");

            table.addView(createFlightColumn("" + R.string.edit_flightNumber,
                    flight.getFlightNumber()));
            table.addView(createFlightColumn("" + R.string.edit_airline,
                    flight.getAirline()));
            table.addView(createFlightColumn("" + R.string.edit_origin,
                    flight.getOrigin()));
            table.addView(createFlightColumn("" + R.string.edit_destination,
                    flight.getDestination()));
            table.addView(createFlightColumn("" + R.string.edit_departure,
                    flight.getDepartureDateTime()));
            table.addView(createFlightColumn("" + R.string.flight_duration,
                    df.format(flight.getDuration())));
            table.addView(createFlightColumn("" + R.string.edit_numSeats,
                    flight.getNumAvailableSeats() + ""));
            table.addView(createFlightColumn("" + R.string.edit_cost,
                    df.format(flight.getCost())));

            // add two empty rows, just to give a little space between flights
            table.addView(createFlightColumn("", ""));
            table.addView(createFlightColumn("", ""));
        }

    }
}
