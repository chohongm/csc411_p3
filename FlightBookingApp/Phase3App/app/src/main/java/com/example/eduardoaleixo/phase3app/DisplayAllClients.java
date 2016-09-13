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

import flightSystem.Client;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

import flightSystem.FlightSystem;
import flightSystem.Itinerary;

/**
 * An activity that displays all the client information a user is allowed to
 * see.
 */
public class DisplayAllClients extends AppCompatActivity {

    /**
     * Initialize the DisplayAllClients layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_display_all_clients);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        final List<Client> allClients;

        //Get the intent that created this activity and it's extras
        Intent intent = getIntent();
        final String curUser = intent.getStringExtra("curUser");

        //If there is no curUser, this is on the admin side of the app, so
        //all clients in the FlightSystem should be dislayed.
        FlightSystem fs = FlightSystem.getInstance();
        if (curUser == null){
            allClients = FlightSystem.getInstance().getClients();

        //If there is a curUser, this is on the client side of the app, so
            //the client can only view their own info
        } else {
            allClients = new ArrayList<Client>();
            allClients.add(fs.searchClient(curUser));
        }

        ArrayList<String> allClientsStr = new ArrayList<>();

        //Build client list of string representations of client objects
        for (Client c : allClients) {
            String[] temp = c.toString().split(",");
            String user = "\nFirst Name: " + temp[1] + "\nLast Name: "
                    + temp[0];

            user = user + "\nBooked Itineraries:";

            for (Itinerary i : c.getBookedItineraries()){
                user = user + "\n" + i.toDisplayString() + "\n";
            }

            allClientsStr.add(user);
        }

        ArrayAdapter<String> arrayAdapter = new ArrayAdapter<String>(this,
                R.layout.client_list_item, android.R.id.text1,
                allClientsStr);

        // Create a new ListView, set the adapter and item click listener
        ListView lv = (ListView) findViewById(R.id.displayAllClients);
        lv.setAdapter(arrayAdapter);
        lv.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Intent intent;

                //Sends admin to a different page where clients are displayed
                intent = new Intent(DisplayAllClients.this,
                        ClientEditClientInfo.class);

                //The client that was clicked
                intent.putExtra("position", position);

                //If there is a not a curUser, the origin is the admin side of
                //the app
                if (curUser == null){
                    intent.putExtra("origin", "admin");

                //If there is a curUser, the origin is the client side of the
                    //app
                } else {
                    intent.putExtra("origin", "client");
                }

                startActivity(intent);

            }


        });
    }

}
