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
import android.widget.Toast;
import org.w3c.dom.Text;

import java.io.IOException;

import flightSystem.Client;
import flightSystem.Flight;
import flightSystem.FlightSystem;

/**
 * An activity allowing users to edit client information.
 */
public class ClientEditClientInfo extends AppCompatActivity {

    //The current user to be edited
    private Client curUser;

    //The origin of the intent
    private String origin;

    /**
     * Initializes the ClientEditClientInfo layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_client_edit_client_info);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        //Get the intent that created this activity
        Intent intent = getIntent();

        //Get the backend
        FlightSystem fs = FlightSystem.getInstance();

        //Get user if it was passed from client
        if (intent.getStringExtra("curUser") != null){
            curUser = fs.searchClient(intent.getStringExtra("curUser"));

        //Get user if it was passed from admin
        } else {
            curUser = fs.getClients().get(intent.getIntExtra("position", 0));
        }

        //Get the origin of the intent
        origin = intent.getStringExtra("origin");

        //Get current user information
        String firstName = curUser.getFirstName();
        String lastName = curUser.getLastName();
        String creditCard = curUser.getCreditCardNumber();
        String address = curUser.getAddress();
        String expiry = curUser.getExpiryDate();
        String email = curUser.getEmail();

        //Set current user information to editText id's for editing page
        EditText display_firstName =
                (EditText) findViewById(R.id.editText_firstname);
        display_firstName.setText(firstName);
        EditText display_lastName =
                (EditText) findViewById(R.id.editText_lastname);
        display_lastName.setText(lastName);
        EditText display_creditCard =
                (EditText) findViewById(R.id.editText_creditcard);
        display_creditCard.setText(creditCard);
        EditText display_address =
                (EditText) findViewById(R.id.editText_address);
        display_address.setText(address);
        EditText display_expiry =
                (EditText) findViewById(R.id.editText_expirydate);
        display_expiry.setText(expiry);
        EditText display_email =
                (EditText) findViewById(R.id.editText_email);
        display_email.setText(email);
    }

    /**
     * Saves client info from on screen when relevant button is pressed.
     * @param view The view of this activity.
     */
    public void saveClientInfo (View view) {

        //Get information from editTexts
        EditText fname = (EditText) findViewById(R.id.editText_firstname);
        EditText lname = (EditText) findViewById(R.id.editText_lastname);
        EditText expiryDate = (EditText) findViewById(R.id.editText_expirydate);
        EditText creditCard = (EditText) findViewById(R.id.editText_creditcard);
        EditText address = (EditText) findViewById(R.id.editText_address);
        EditText email = (EditText) findViewById(R.id.editText_email);

        if (curUser != null){

            curUser.setCreditCardNumber(creditCard.getText().toString());
            curUser.setEmail(email.getText().toString());
            curUser.setFirstName(fname.getText().toString());
            curUser.setLastName(lname.getText().toString());
            curUser.setLastName(address.getText().toString());
            curUser.setExpiryDate(expiryDate.getText().toString());
            curUser.setCreditCardNumber(creditCard.getText().toString());
            curUser.setAddress(address.getText().toString());

            //Save new information to FlightSystem
            FlightSystem fs = FlightSystem.getInstance();
            try {
                fs.save(this.getApplicationContext().getFilesDir().toString());
                Toast.makeText(this, R.string.information_saved,
                        Toast.LENGTH_SHORT).show();
            } catch (IOException e) {
                Toast.makeText(this, R.string.information_not_saved,
                        Toast.LENGTH_SHORT).show();
            }

            //Go back to menu, pass the curUser in intent
            if (origin.equals("admin")) {
                Intent new_intent = new Intent(this, AdminMenu.class);
                startActivity(new_intent);
            }

            else if (origin.equals("client")){
                Intent new_intent = new Intent(this, ClientMenu.class);

                //Put new modified user object
                new_intent.putExtra("curUser", curUser.getEmail());
                startActivity(new_intent);
            }
        }
    }
}
