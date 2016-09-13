package com.example.eduardoaleixo.phase3app;

import android.content.Intent;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import flightSystem.Flight;
import flightSystem.FlightSystem;

/**
 * An activity that allows admins to edit flights.
 */
public class EditFlightActivity extends AppCompatActivity {

    //The flight to be edited
    private Flight oldFlight;

    /**
     * Initializes the EditFligthActivity layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_edit_flight);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        //Get the flight to be edited
        Intent intent = getIntent();
        final Flight flight = (Flight) intent.getSerializableExtra("flight");
        this.oldFlight = flight;

        init(flight);
    }

    /**
     * Display flight information on the screen.
     * @param flight The flight who's information will be displayed.
     */
    private void init(Flight flight){
        ((EditText) findViewById(R.id.editText_airline)).setText(
                flight.getAirline());
        ((EditText) findViewById(R.id.editText_flightNumber)).setText(
                flight.getFlightNumber() + "");
        ((EditText) findViewById(R.id.editText_origin)).setText(
                flight.getOrigin());
        ((EditText) findViewById(R.id.editText_destination)).setText(
                flight.getDestination());
        ((EditText) findViewById(R.id.editText_numSeats)).setText(
                flight.getNumSeats() + "");
        ((EditText) findViewById(R.id.editText_departure)).setText(
                flight.getDepartureDateTime());
        ((EditText) findViewById(R.id.editText_cost)).setText(
                flight.getCost() + "");
    }

    /**
     * Edits flight information based on what's on the screen when the relevant
     * button is pressed.
     * @param v The view of this activity.
     */
    public void editFlight(View v){
        TextView status = (TextView) findViewById(R.id.textView_status);
        Flight newFlight =
                FlightSystem.getInstance().searchOneFlight(this.oldFlight);

        //In case of an error
        if(newFlight == null){
            status.setText(R.string.flight_not_found);
            return ;
        }

        //Edit flight information
        try{
            newFlight.setAirline(
                    ((EditText) findViewById(R.id.editText_airline)).
                            getText().toString()
            );
            newFlight.setFlightNumber(
                    ((EditText) findViewById(R.id.editText_flightNumber)).
                            getText().toString()
            );
            newFlight.setArrivalDateTime(
                    ((EditText) findViewById(R.id.editText_departure)).
                            getText().toString()
            );
            newFlight.setDestination(
                    ((EditText) findViewById(R.id.editText_destination)).
                            getText().toString()
            );
            newFlight.setOrigin(
                    ((EditText) findViewById(R.id.editText_origin)).
                            getText().toString()
            );
            newFlight.setNumSeats(
                    Integer.parseInt(
                            (((EditText) findViewById(R.id.editText_numSeats)).
                                    getText().toString()).trim())
            );

            newFlight.setCost(
                    Double.parseDouble(
                            ((EditText) findViewById(R.id.editText_cost)).
                                    getText().toString())
            );

            //Save the new flight information
            FlightSystem.getInstance().save(
                    this.getApplicationContext().getFilesDir().toString());

            status.setText(R.string.success_flight_edited);
        } catch (Exception e){
             status.setText(e.getMessage());
        }


    }

}
