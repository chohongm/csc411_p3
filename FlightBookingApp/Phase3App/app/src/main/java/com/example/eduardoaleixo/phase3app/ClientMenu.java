package com.example.eduardoaleixo.phase3app;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;

import flightSystem.Client;

/**
 * An activity for a client's main menu.
 */
public class ClientMenu extends AppCompatActivity {

    //The email of the user currently using the app
    private String curUser;

    /**
     * Initializes the Client Menu layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_client_menu);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        //Get the intent that created this activity
        Intent intent = getIntent();

        //Get the email of the client currently using the app
        curUser = intent.getStringExtra("curUser");
    }

    /**
     * Starts SearchFlightActivity activity when the relevant button is pressed.
     * @param view The view of this activity.
     */
    public void searchFlightActivity(View view){
        Intent intent = new Intent(this, SearchFlightActivity.class);

        //Pass the current Client's email with the intent
        intent.putExtra("curUser", curUser);
        startActivity(intent);
    }

    /**
     * Starts ItinerarySearch activity when the relevant button is pressed.
     * @param view The view of this activity.
     */
    public void searchItinerariesActivity(View view){
        Intent intent = new Intent(this, ItinerarySearch.class);

        //Pass the current Client's email with the intent
        intent.putExtra("curUser", curUser);
        startActivity(intent);
    }

    /**
     * Starts DisplayAllClients activity when the relevant button is pressed.
     * @param view The view of this activity.
     */
    public void editClientInfo(View view) {
        Intent intent = new Intent(this, DisplayAllClients.class);

        //Pass the current Client's email with the intent
        intent.putExtra("curUser", curUser);

        //The origin of this intent is on the client side
        intent.putExtra("origin", "client");
        startActivity(intent);
    }


}
