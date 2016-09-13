package com.example.eduardoaleixo.phase3app;

import android.content.Intent;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.ViewFlipper;

/**
 * An activity for the admin's main menu.
 */
public class AdminMenu extends AppCompatActivity {

    /**
     * Initializes the AdminMenu layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_admin_menu);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

    }

    /**
     * Starts SearchFlightActivity activity when the relevant button is pressed.
     * @param view The view of this activity.
     */
    public void goToSearchForFlights(View view){
        Intent intent = new Intent(this, SearchFlightActivity.class);

        startActivity(intent);
    }

    /**
     * Starts LoadFlights activity when the relevant button is pressed.
     * @param view The view of this activity.
     */
    public void goToLoadFlights(View view){
        Intent intent = new Intent(this, LoadFlights.class);

        startActivity(intent);
    }

    /**
     * Starts ViewFlights activity when the relevant button is pressed.
     * @param view The view of this activity.
     */
    public void goToViewFlights(View view){
        Intent intent = new Intent(this, ViewFlightsActivity.class);

        startActivity(intent);
    }

    /**
     * Starts ItinerarySearch activity when the relevant button is pressed.
     * @param view The view of this activity.
     */
    public void searchItineraries(View view){
        Intent intent = new Intent(this, ItinerarySearch.class);

        startActivity(intent);
    }

    /**
     * Starts UploadClientInfo activity when the relevant button is pressed.
     * @param view The view of this activity.
     */
    public void goToUploadClientInfo(View view){
        Intent intent = new Intent(this, UploadClientInfo.class);

        startActivity(intent);
    }

    /**
     * Starts DisplayAllClients activity when the relevant button is pressed.
     * @param view The view of this activity.
     */
    public void displayAllClients(View view) {
        Intent intent = new Intent(this, DisplayAllClients.class);

        startActivity(intent);
    }
}
