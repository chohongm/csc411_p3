package com.example.eduardoaleixo.phase3app;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Scanner;

import flightSystem.Client;
import flightSystem.FlightSystem;

/**
 * The main activity. Allows users to login and determines if they are an
 * admin or a client.
 */
public class LoginActivity extends AppCompatActivity {

    //Holds valid username/password combinations
    private HashMap<String, String> clientPasswords = new HashMap<String, String>();
    private HashMap<String, String> adminPasswords = new HashMap<String, String>();

    //The backend FlightSystem
    private FlightSystem fs;

    /**
     * Initializes the LoginActivity layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        //Sets up the singleton FlightSystem instance.
        // Loads object if there is one to load.
        FlightSystem.load(
                this.getApplicationContext().getFilesDir().toString());
        fs = FlightSystem.getInstance();

        //Hardcodes in a client so that the client/client username/password
        //combination can be used on first run
        if (fs.searchClient("client") == null){
            fs.addClient("Doe","John","client","1 Main St","1234","10/17");
        }
    }


    public void login(View view) {
        EditText username = (EditText) findViewById(R.id.editText_login);
        EditText password = (EditText) findViewById(R.id.editText_password);
        TextView error_message =
                (TextView) findViewById(R.id.textView_error_message);
        Intent intent;

        //Reads the password.txt file for log in authentication
        readPasswordsFile();

        String user = username.getText().toString();
        String pass = password.getText().toString();

        //Validates username password combination
        if (adminPasswords.containsKey(user) &&
                adminPasswords.get(user).equals(pass) ){

            intent = new Intent(this,AdminMenu.class);
            startActivity(intent);

        } else if (clientPasswords.containsKey(user) &&
                clientPasswords.get(user).equals(pass)) {

            //Checks for client in FlightSystem database.
            Client currentClient = fs.searchClient(user);

            //Go to client menu if the client exists in the database
            if (currentClient != null) {
                intent = new Intent(this, ClientMenu.class);
                intent.putExtra("curUser", user);
                startActivity(intent);
            } else {
                error_message.setText(R.string.verify_in_database);
            }

        } else {
            error_message.setText(R.string.invalid_username_password);
        }
    }

    private void readPasswordsFile(){
        String file = this.getApplicationContext().getFilesDir().toString() +
                "/passwords.txt";

        try {
            Scanner sc = new Scanner(new File(file));

            //Temporary variables for reading the file.
            String nextLine;
            String [] temp;

            //Cycle through each line of the file
            while (sc.hasNextLine()){
                nextLine = sc.nextLine();
                temp = nextLine.split(",");

                if (temp[2].equals("Y")){
                    adminPasswords.put(temp[0], temp[1]);
                } else {
                    clientPasswords.put(temp[0], temp[1]);
                }
            }
        } catch (FileNotFoundException f){
            TextView error_message
                    = (TextView) findViewById(R.id.textView_error_message);
            error_message.setText(R.string.passwords_file_not_found);
        }
    }
}
