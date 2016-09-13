package com.example.eduardoaleixo.phase3app;

import android.app.DatePickerDialog;
import android.app.Dialog;
import android.content.Intent;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.Menu;
import android.view.View;
import android.widget.DatePicker;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.List;
import java.util.Scanner;

import flightSystem.Client;
import flightSystem.FileFormatException;
import flightSystem.Flight;
import flightSystem.FlightSystem;
import flightSystem.Itinerary;

/**
 * An activity that allows for users to serach for itineraries.
 */
public class ItinerarySearch extends AppCompatActivity {

    // singleton instance of FlightSystem
    FlightSystem fs;
    // intent instance
    Intent intent;
    // current user instance using this activity
    String curUser;
    // calendar for displaying and selecting departure date.
    private Calendar calendar;
    // a textView referring to departure date.
    private TextView dateView;
    // year, month, day of departure date to be selected/displayed
    private int year, month, day;
    // departure date in string format.
    private String deptDate;
    // a string referring to selected sorting type of searched itineraries.
    private String sortBy;

    @Override
    /**
     * Initializes the ItinerarySearch layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_itinerary_search);

        dateView = (TextView) findViewById(R.id.deptDate);
        calendar = Calendar.getInstance();
        year = calendar.get(Calendar.YEAR);
        month = calendar.get(Calendar.MONTH);
        day = calendar.get(Calendar.DAY_OF_MONTH);
        deptDate = "" + year + "-" + month + "-" + day;
        showDate(year, month + 1, day);
        // by default sort by cost since sortByCost radio button is checked.
        sortBy = "cost";

        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        fs = FlightSystem.getInstance();

        intent = getIntent();
        curUser = intent.getStringExtra("curUser");
    }

    /**
     * Sets sorting choice based on checked radio button.
     * @param view selected radio button representing a sorting type of
     *             itinerary search.
     */
    public void onRadioButtonClicked(View view) {
        // Is the button now checked?
        boolean checked = ((RadioButton) view).isChecked();
        // Check which radio button was clicked
        switch(view.getId()) {
            case R.id.sortByCost:
                if (checked)
                    sortBy = "cost";
                    break;
            case R.id.sortByTime:
                if (checked)
                    sortBy = "time";
                    break;
        }
    }

    @SuppressWarnings("deprecation")
    /**
     * Picks departure date.
     * @param view departure date button.
     */
    public void pickDeptDate(View view) {
        showDialog(999);
        Toast.makeText(getApplicationContext(), "ca",
                Toast.LENGTH_SHORT).show();
    }

    @Override
    /**
     * Returns DatePickerDialog given OnDateSetListener, year, month,
     * day if given id is 999, and null if not 999.
     * @param id id that refers to a certain dialog to be returned.
     */
    protected Dialog onCreateDialog(int id) {

        if (id == 999) {
            return new DatePickerDialog(this, myDateListener, year, month, day);
        }
        return null;
    }

    private DatePickerDialog.OnDateSetListener myDateListener
            = new DatePickerDialog.OnDateSetListener() {
        @Override
        public void onDateSet(DatePicker arg0, int year, int month, int day) {

            showDate(year, month+1, day);
        }
    };

    /**
     * Displays the selected departure date in text view.
     * @param year year of departure date.
     * @param month month of departure date.
     * @param day day of departure date,
     */
    private void showDate(int year, int month, int day) {
        String m = "" + month;
        String d = "" + day;
        if (m.length() == 1) {
            m = "0" + m;
        }
        if (d.length() == 1) {
            d = "0" + d;
        }
        deptDate = "" + year + "-" + m + "-" + d;
        dateView.setText(deptDate);
    }

    @Override
    /**
     * Displays menu bar.
     * @param menu the menu bar to be displayed.
     */
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    /**
     * Searches and sends itineraries sorted by selected sort type given origin,
     * destination and chosen departure date.
     * @param view search button
     * @throws ParseException
     * @throws FileNotFoundException
     * @throws FileFormatException
     */
    public void searchItineraries(View view)
            throws ParseException, FileNotFoundException, FileFormatException {
        // Specifies the next Activity to move to: DisplayActivity.
        intent = new Intent(this, SearchedItinerariesDisplay.class);
        intent.putExtra("curUser", curUser);

        // Gets the data from the EditText field.
        EditText origin = (EditText) findViewById(R.id.origin);
        EditText destination = (EditText) findViewById(R.id.destination);


        fs.searchForItineraries(origin.getText().toString(),
                destination.getText().toString(), deptDate);
        if (sortBy.equals("cost")) {
            fs.sortItinerariesByCost();
        } else {
            fs.sortItinerariesByTime();
        }
        // Passes the String data to DisplaySearchedItineraries.
        startActivity(intent); // Starts DisplayActivity.x
    }
}
